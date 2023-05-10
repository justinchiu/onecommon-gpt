#!/bin/bash

tmux new-session -s "oc" -n "root" "tmux source-file tmux/session"
