#!/bin/bash

set -e

usage() {
    echo "Usage: $0 train [model|config_path] [train_options...]"
    echo
    echo "Built-in models:"
    echo "  davit    -> configs/davit/davit-eye.py"
    echo "  resnet   -> configs/resnet/resnet50-eye.py"
    echo
    echo "Examples:"
    echo "  $0 train"
    echo "  $0 train resnet"
}

resolve_config() {
    local model="$1"
    case "$model" in
        ""|"davit"|"davit-eye")
            echo "configs/davit/davit-eye.py"
            ;;
        "resnet"|"resnet50"|"resnet50-eye")
            echo "configs/resnet/resnet50-eye.py"
            ;;
        *)
            if [ -f "$model" ]; then
                echo "$model"
            else
                return 1
            fi
            ;;
    esac
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

mode="$1"
shift

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    model="$1"
    shift
else
    model="davit"
fi

if ! config_path="$(resolve_config "$model")"; then
    echo "Unknown model or config path: $model"
    echo
    usage
    exit 1
fi

case "$mode" in
    train)
        echo "Starting training with $config_path ..."
        python3 tools/train.py "$config_path" "$@"
        ;;
    *)
        echo "Invalid mode: $mode"
        echo
        usage
        exit 1
        ;;
esac
