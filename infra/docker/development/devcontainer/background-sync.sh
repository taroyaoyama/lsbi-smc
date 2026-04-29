#!/bin/bash
# node ユーザーとして実行すること（sudo で呼ばないこと）
echo "Background sync started..."
cd /workspace/lsbi-smc && uv sync --frozen

echo "Background sync finished."