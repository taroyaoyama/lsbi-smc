#!/usr/bin/env python3
"""TLS ClientHello SNI extractor — reads pcap stream from stdin (tcpdump -w -)"""
import sys
import struct


def extract_sni(payload):
    try:
        if len(payload) < 5 or payload[0] != 0x16:
            return None
        rec_len = struct.unpack('>H', payload[3:5])[0]
        if len(payload) < 5 + rec_len:
            return None
        hs = payload[5:5 + rec_len]
        if not hs or hs[0] != 0x01:
            return None
        # handshake header(4) + version(2) + random(32)
        off = 38
        if off >= len(hs):
            return None
        sess_len = hs[off]
        off += 1 + sess_len
        if off + 2 > len(hs):
            return None
        cipher_len = struct.unpack('>H', hs[off:off + 2])[0]
        off += 2 + cipher_len
        if off >= len(hs):
            return None
        comp_len = hs[off]
        off += 1 + comp_len
        if off + 2 > len(hs):
            return None
        ext_end = off + 2 + struct.unpack('>H', hs[off:off + 2])[0]
        off += 2
        while off + 4 <= ext_end:
            ext_t = struct.unpack('>H', hs[off:off + 2])[0]
            ext_l = struct.unpack('>H', hs[off + 2:off + 4])[0]
            off += 4
            if ext_t == 0 and off + 5 <= ext_end:  # SNI extension (type=0)
                name_len = struct.unpack('>H', hs[off + 3:off + 5])[0]
                return hs[off + 5:off + 5 + name_len].decode('ascii', errors='ignore')
            off += ext_l
    except Exception:
        pass
    return None


def main():
    stdin = sys.stdin.buffer

    # pcap global header (24 bytes) — contains datalink type
    hdr = stdin.read(24)
    if len(hdr) < 24:
        return
    dl_type = struct.unpack('<I', hdr[20:24])[0]
    # Ethernet=1 (14 bytes), Linux cooked capture=113 (16 bytes)
    link_len = {1: 14, 113: 16}.get(dl_type, 14)

    while True:
        rec = stdin.read(16)
        if len(rec) < 16:
            break
        cap_len = struct.unpack('<I', rec[8:12])[0]
        pkt = stdin.read(cap_len)
        if len(pkt) < cap_len:
            break

        ip_start = link_len
        if ip_start + 20 > len(pkt):
            continue
        # IPv4 only, TCP only
        if pkt[ip_start] >> 4 != 4 or pkt[ip_start + 9] != 6:
            continue

        ip_hl = (pkt[ip_start] & 0x0f) * 4
        tcp_start = ip_start + ip_hl
        if tcp_start + 20 > len(pkt):
            continue
        tcp_hl = ((pkt[tcp_start + 12] >> 4) & 0xf) * 4

        sni = extract_sni(pkt[tcp_start + tcp_hl:])
        if sni:
            print(sni, flush=True)


if __name__ == '__main__':
    main()
