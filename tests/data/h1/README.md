# H1 model for headless tests

This fixture retains the joints, limits, frames, masses, and inertias from
`HumanoidAssets/UnitreeH1/robot.urdf` at commit
`ef17d6ec585b965cfd9a469a3ef28f60705db7a0` of
<https://github.com/chase6305/HumanoidAssets>. That model derives from
Unitree Robotics' `unitree_ros/robots/h1_description` model.

The source file's SHA-256 is
`e03c4fa93f308b5e14a489a5e9dbdfdbe9a1bd81e5c5aaa24e6d05614a581742`.
Only the MuJoCo compiler settings and visual/collision geometry were removed;
the headless control tests do not load geometry. No mesh downloads are needed.
The upstream BSD 3-Clause license is retained in [LICENSE](LICENSE).

The native CI job copies this file to the examples' existing sibling-model
location before running the suite. H1 physics and kinematic checks then run
alongside the self-contained URDF tests instead of being skipped.
