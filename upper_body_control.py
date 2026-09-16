#!/usr/bin/env python3
"""
Upper body control example.

Usage:
  ros2 run py_examples upper_body_control <mode>

  mode:
    head    — head center pose,          requires HEAD_ONLY mode
    claw    — claw half-open,            requires UPPERBODY_REMOTE_SPLIT mode
    joint   — dexterous hand half-open,  requires UPPERBODY_REMOTE_SPLIT mode
    gesture — dexterous hand gesture 1,  requires UPPERBODY_REMOTE_SPLIT mode

Switch to the desired MC mode before running, e.g. for head:
  ros2 run py_examples set_mc_action SD     # → STAND_DEFAULT
  ros2 run py_examples set_mc_action HO     # → HEAD_ONLY
or for claw / joint / gesture:
  ros2 run py_examples set_mc_action URS    # → UPPERBODY_REMOTE_SPLIT
"""

import sys
import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import UpperBodyCommandArray, MessageHeader

# hand_sub_mode, head_pos, arm_pos, hand_pos
MODES = {
    # head_yaw=0, head_pitch=0 (center)
    'head':    (0, [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                []),
    # left_open=0.5, right_open=0.5
    'claw':    (UpperBodyCommandArray.HAND_CLAW_OPEN_CLOSE,  # claw (1)
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5]),
    # left 10 joints + right 10 joints, in rad, all half-open
    # (each joint at half its URDF range limit). Right is the mirror of left.
    # Joint order: [thumb_roll, thumb_abad, thumb_mcp, index_abad, index_pip,
    #               middle_pip, ring_abad, ring_pip, pinky_abad, pinky_pip]
    'joint':   (UpperBodyCommandArray.HAND_DEXTEROUS_JOINT,  # dexterous joint (2)
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.6, 0.8, -0.4, 0.1, 0.7, 0.7, -0.1, 0.7, -0.1, 0.7,
                 0.6, -0.8, 0.4, -0.1, 0.7, 0.7, 0.1, 0.7, 0.1, 0.7]),
    # left_gesture=1, left_open=1.0, right_gesture=1, right_open=1.0
    'gesture': (UpperBodyCommandArray.HAND_DEXTEROUS_GESTURE,  # dexterous gesture (3)
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0]),
}


class UpperBodyControlNode(Node):
    def __init__(self, mode: str):
        super().__init__('upper_body_control')
        self.hand_sub_mode, self.head_pos, self.arm_pos, self.hand_pos = MODES[mode]
        self.pub = self.create_publisher(
            UpperBodyCommandArray, '/mc/upper_body_command', 10)
        self.timer = self.create_timer(0.02, self.publish)  # 50 Hz
        self._seq = 0
        self.get_logger().info(
            f'mode={mode}  hand_sub_mode={self.hand_sub_mode}')

    def publish(self):
        msg = UpperBodyCommandArray()
        now = self.get_clock().now()
        msg.header = MessageHeader()
        msg.header.stamp.sec = now.nanoseconds // 1_000_000_000
        msg.header.stamp.nanosec = now.nanoseconds % 1_000_000_000
        msg.header.frame_id = 'mc_upper_body'
        msg.header.sequence = self._seq
        self._seq += 1
        msg.source = 'upper_body_example'
        msg.hand_sub_mode = self.hand_sub_mode
        msg.head_pos = self.head_pos
        msg.arm_pos = self.arm_pos
        msg.hand_pos = self.hand_pos
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    argv = sys.argv[1:]
    if not argv or argv[0] not in MODES:
        print(f'Usage: upper_body_control <{"│".join(MODES)}>')
        rclpy.shutdown()
        return

    node = UpperBodyControlNode(argv[0])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
