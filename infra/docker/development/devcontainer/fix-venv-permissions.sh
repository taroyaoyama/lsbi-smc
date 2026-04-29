#!/bin/bash
# .venv の ownership を node:node に修正（root で実行される）
chown -R node:node /workspace/lsbi-smc/.venv
