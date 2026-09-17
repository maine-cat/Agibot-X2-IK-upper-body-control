# V2.1 desktop bridge protocol

Robot entry (paths are customer supplied):

```bash
cd /path/to/robot/runtime
X2IK_CONFIG=/path/to/robot/x2ik.conf python3 -m x2ik mdi --stdio --tcp-mode gripper
```

MDI motion is supported only while the robot is already in URS (`UPPERBODY_REMOTE_SPLIT`).
Customers switch to URS through the robot's existing controls. MDI has no mode-switch operation,
button or environment-variable override. Startup is read-only and creates no motion publisher.
The product version is V2.1, package/artifact version `2.1.0`; this release is developed on branch `V2.0`.

## TCP session configuration

`--tcp-mode none|hand|gripper|custom` defaults to `none` (wrist origin).
`custom` also requires `--tcp-file /absolute/robot/path/tool.json`.
The desktop passes the selected mode and path as quoted process arguments over SSH;
for a real connection the path is on the robot. There is no file-upload request.
Only local `--demo` interprets the custom path on the desktop computer.

The selected tool is fixed for the session. Requests containing `tcp_mode`, `tcp_file`
or `tcp_config` are rejected; disconnect and create a new session to change tools.
`hand` is an estimated cup grasp center, `gripper` a nominal URDF center. Neither
operates fingers or adjusts payload/compensation. Full transforms and calibration
instructions are in the customer TCP guide.

## Requests and results

Requests are JSON Lines objects `{id, op}`. Supported operations:
`heartbeat`, `arm`, `disarm`, `mdi`, `home`, `state`, `close`.

MDI fields:

| Field | Values / meaning |
| --- | --- |
| `mode` | `xyz`, `pose`, `d`, `R`, `t`, `rpy`, `j` |
| `side` | `left` / `right` |
| `values` | Numeric target list; Cartesian targets refer to selected TCP |
| `duration` / `settle` | Seconds |
| `preview` | Boolean; previews send no motion and disarm an existing hold |

Position values are metres; joint and rotation angles are degrees.
`j` and HOME remain joint targets regardless of tool. HOME always affects both arms
and accepts optional duration/settle/preview. `op: action` is rejected, including
old-client requests and requests with `force`; `action` in status is read-only feedback.

Results: `{type:'result', id, ok, message}`. Events: `{type:'event', message}`.

## State and 3D geometry

Unsolicited state is emitted at approximately 5 Hz:

```text
{
  type: 'state', demo, connected, armed, busy, action, fresh,
  urs_confirmed, command_endpoint_owned, tcp_mode,
  arms: {
    left:  {q_deg, xyz, rpy_deg, points, tcp, link_transforms},
    right: {q_deg, xyz, rpy_deg, points, tcp, link_transforms}
  }
}
```

`xyz/rpy_deg` are selected TCP pose in `torso_link`, calculated from fresh joint
feedback. `tcp` contains `mode/name/frame/translation_m/rotation_matrix/source`
and `estimated/description`. Fixed tool transforms map TCP vectors into that
side's `wrist_roll_link`. There is no external tracking or grasp-center sensing.

`points` contains the seven joint origins followed by the selected TCP point,
in torso-frame metres. `link_transforms` contains seven objects `{xyz, rotation}`
in shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll order. Each rotation is
post-joint, maps the corresponding link frame into torso, and is a full 3×3 matrix.
The client uses these poses with STL-derived decimated arm meshes. These visual
meshes are not STEP solids or a collision model. An older state without
link_transforms may use the validated local chain and q_deg, explicitly labelled
local FK. Invalid supplied transforms or unavailable assets cause skeleton fallback.
Only the 14 arm links are drawn; hands/grippers are represented by their TCP point
and axes rather than tool meshes.

Invalid joint feedback clears `arms` and sets `fresh=false`. `urs_confirmed`
requires mode feedback no older than 3 seconds. The V2.1 desktop also requires the
backend `tcp_mode` to match its requested mode before sending can be enabled;
an older/unconfirmed backend must not be treated as if it has the requested TCP.
The bridge emits this top-level session mode even when `arms` is empty; when
feedback is available, both `arms.left.tcp.mode` and `arms.right.tcp.mode` equal it.

## Sending lifecycle

Heartbeat is required within 2 seconds while armed. EOF/disarm/close/heartbeat
expiry or loss of confirmed URS stops sends without HOME or mode switching.
`arm` requires fresh joint feedback and confirmed URS. First explicit motion
creates the ROS publisher/session and performs prechecks; motion independently
rechecks URS. Outside URS, feedback and previews remain available. Restoring URS
requires explicit re-enabling. Previews validate target IK, not collisions or
complete path clearance. Busy commands are rejected, not queued for later motion.
