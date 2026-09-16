# Implemented backend protocol

Robot entry: `cd /home/agi/x2ik/runtime && X2IK_CONFIG=/home/agi/x2ik/x2ik.conf python3 -m x2ik mdi --stdio`.

MDI motion is supported only while the robot is already in URS (`UPPERBODY_REMOTE_SPLIT`). Customers must switch to URS through the robot's existing controls before enabling sending. MDI provides no mode switching operation or UI; environment variables cannot enable one.

Requests: `{id, op}`. `op`: heartbeat, arm, disarm, mdi, home, state, close.
MDI extra keys: `mode` xyz/pose/d/R/t/rpy/j, `side` left/right, `values` list, `duration` seconds, `settle` seconds, `preview` boolean. Position values metres; joint/rotation angles degrees.
`op: action` is rejected, including requests from older clients and requests with `force`. `action` in status messages is read-only robot feedback.
Home always both arms, optional duration/settle/preview.

Unsolicited status 5Hz: `{type:'state', demo, connected, armed, busy, action, fresh, urs_confirmed, command_endpoint_owned, arms:{left:{q_deg, xyz, rpy_deg, points}, right:{...}}}`. `points` is array of [x,y,z] metres in torso frame. Invalid feedback sets arms={} and fresh=false. `urs_confirmed` also requires mode feedback no older than 3 seconds. Results `{type:'result', id, ok, message}`. Events `{type:'event', message}`.

Heartbeat required <2s while armed. EOF/disarm/close/heartbeat expiry or loss of confirmed URS stops sends; no HOME or state transition. Startup reads feedback only, zero command publisher. `arm` requires fresh joint feedback and confirmed URS. The first explicit motion command creates the ROS publisher/session and performs read-only prechecks. Motion independently rechecks URS even after `arm`. Outside URS, joint/pose feedback and previews remain available, but sending is disabled. Once URS is restored, customers must explicitly enable sending again. Previews are strict target IK (no collision checking), no sending; preview disarms any existing hold session.
