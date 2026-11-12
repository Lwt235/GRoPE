"""Reuse resource admission/yielding with isolated experiment status paths."""
from experiments import native_receiver_launch as launcher
from experiments.native_receiver_view_data import OUT

if __name__=='__main__':
    launcher.OUT=OUT
    launcher.main()
