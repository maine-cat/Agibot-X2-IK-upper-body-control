#!/bin/sh
# Install the x2ik module and prepare this robot's ROS / AimDK environment first.
# Set X2IK_CONFIG to this robot's own configuration path.
# Default: help only. --dry: ROS feedback subscription, no motion publication.
# --execute: live MDI control, including continuous holding in an online session.
# Optional TCP: --tcp-mode none|hand|gripper|custom [--tcp-file /robot/tool.json].
# TCP changes the target frame, not finger/gripper actuation. URS is required.
set -eu

case "${1:-}" in
    ""|--help|-h)
        exec python3 -m x2ik mdi --help
        ;;
    --dry)
        shift
        exec python3 -m x2ik mdi --dry "$@"
        ;;
    --execute)
        shift
        exec python3 -m x2ik mdi "$@"
        ;;
    *)
        printf '%s\n' 'Usage: sh mdi_minimal.sh [--dry|--execute] [--side left|right] [--tcp-mode none|hand|gripper|custom] [--tcp-file /robot/tool.json] [--duration 8] [--settle 2]' >&2
        exit 2
        ;;
esac
