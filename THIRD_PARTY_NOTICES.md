# Model and dependency provenance

`x2_ultra.urdf` and `x2_ultra.xml` are the upstream X2 Ultra model assets already
present in this project's GitLab `v1.0` branch. The accompanying upstream notice
is retained as `x2_urdf_upstream_README.md`; it declares Mulan PSL v2 for the model.
Visual mesh files and the complete upstream SDK are not distributed in this source export.
The shipped FK/IK and desktop skeleton view use the model's kinematic parameters.

Runtime dependencies include Python and NumPy; online robot operation additionally
requires the operator's existing ROS 2 / AimDK message installation. Desktop builds
use PyQt5/Qt, OpenSSH, PyInstaller and AppImage tooling. Their licenses apply to their
respective components; the delivery builder includes available distribution notices.

This notice does not grant a new license to third-party components or to the
project's own code. Preserve the upstream notices when redistributing model files.
