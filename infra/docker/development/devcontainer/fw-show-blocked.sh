#!/bin/bash
# ブロックされた通信をIPレベル・ドメインレベルで表示するヘルパー
# 使い方:
#   fw-show-blocked.sh           通常表示
#   fw-show-blocked.sh --watch   TCP接続試行をリアルタイム表示

set -euo pipefail

WATCH=false
[[ "${1:-}" == "--watch" ]] && WATCH=true

# DNS ログから IP → ホスト名 のマッピングを返す
lookup_hostname() {
    local ip="$1"
    # "hostname. A IP" 形式から hostname を抽出
    grep -oP "\S+(?=\.\s+A\s+${ip//./\\.}([^0-9]|$))" \
        /var/log/firewall/dns.log 2>/dev/null \
        | sort -u | head -3 | tr '\n' ',' | sed 's/,$//' \
        || true
}

print_tcp_attempts() {
    echo "=== TCP接続試行（新規接続 SYNパケット） ==="
    echo "  ※ iptables LOG は Docker コンテナ内では使用不可のため tcpdump で代替"
    echo "  ※ tcpdump はアウトバウンドを NIC 送出後に捕捉 → REJECT されたパケットは記録されない（許可された接続のみ）"

    if [ ! -s /var/log/firewall/tcp-attempts.log ]; then
        echo "  （ログなし — コンテナ起動直後または接続試行がまだない場合）"
        return
    fi

    # tcpdump -i any 形式: "timestamp  interface  direction  IP  src.port > dst.port: Flags [S]..."
    # フィールド: $1=time $2=iface $3=In/Out $4=IP $5=src.port $6=> $7=dst.port
    awk '$6 == ">" {
        dst = $7
        sub(/:$/, "", dst)
        n = split(dst, p, ".")
        port = p[n]
        ip = p[1]"."p[2]"."p[3]"."p[4]
        print ip, port
    }' /var/log/firewall/tcp-attempts.log \
        | sort | uniq -c | sort -rn \
        | while read -r count ip port; do
            host=$(lookup_hostname "$ip")
            printf "  %4s回  %-16s :%-6s  %s\n" \
                "$count" "$ip" "$port" "${host:-(DNS未解決)}"
        done
}

# DNS解決済みだがTCP接続が確認されていないIP = ブロックされた可能性が高い
print_likely_blocked() {
    echo ""
    echo "=== ブロックされた接続の推定（DNS解決済み & TCP未確認） ==="
    echo "  ※ DNS解決されたがTCP SYNが記録されていない宛先 → iptablesにブロックされた可能性"
    echo "  ※ アプリが接続を試みなかった場合もあるため「推定」"

    if [ ! -s /var/log/firewall/dns.log ]; then
        echo "  （DNSログなし）"
        return
    fi

    # tcp-attempts.log から接続済みIPをセットとして収集
    local attempted_ips
    attempted_ips=$(awk '$6 == ">" {
        dst = $7; sub(/:$/, "", dst)
        n = split(dst, p, ".")
        print p[1]"."p[2]"."p[3]"."p[4]
    }' /var/log/firewall/tcp-attempts.log 2>/dev/null | sort -u)

    # dns.log から解決済みIPを抽出
    local dns_ips
    dns_ips=$(grep -oP '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' \
        /var/log/firewall/dns.log 2>/dev/null \
        | grep -vE '^(127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' \
        | sort -u)

    if [ -z "$dns_ips" ]; then
        echo "  （DNS解決済みIPなし）"
        return
    fi

    local found=false
    while read -r ip; do
        if ! echo "$attempted_ips" | grep -qF "$ip"; then
            host=$(lookup_hostname "$ip")
            printf "  BLOCKED?  %-16s  %s\n" "$ip" "${host:-(ホスト名不明)}"
            found=true
        fi
    done <<< "$dns_ips"

    if ! $found; then
        echo "  （なし — DNS解決された外部IPはすべてTCP接続確認済み）"
    fi
}

print_sni() {
    echo ""
    echo "=== HTTPS接続先ホスト名（TLS SNI） ==="
    echo "  ※ パスはHTTPS暗号化のため取得不可"
    if [ -s /var/log/firewall/sni.log ]; then
        sort -u /var/log/firewall/sni.log | sed 's/^/  /'
    else
        echo "  （ログなし）"
    fi
}

print_dns_log() {
    echo ""
    echo "=== DNS解決ログ（接続試行されたホスト名 → IP） ==="
    echo "  ※ 許可・ブロック問わずDNS解決されたもの全件"
    if [ -s /var/log/firewall/dns.log ]; then
        grep -oP '\S+\.\s+A\s+[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' \
            /var/log/firewall/dns.log 2>/dev/null \
            | sort -u | sed 's/^/  /' \
            || echo "  （A レコードなし）"
    else
        echo "  （ログなし）"
    fi
}

if $WATCH; then
    echo "リアルタイム監視モード（Ctrl+C で終了）..."
    echo "---"
    tail -f /var/log/firewall/tcp-attempts.log 2>/dev/null \
        | awk '$6 == ">" {
            dst = $7; sub(/:$/, "", dst)
            n = split(dst, p, "."); port = p[n]
            ip = p[1]"."p[2]"."p[3]"."p[4]
            printf "ATTEMPT  %s -> %s:%s\n", $5, ip, port
            fflush()
        }'
else
    print_tcp_attempts
    print_likely_blocked
    print_sni
    print_dns_log
fi
