"""Register the shared hardware fakes as pytest fixtures."""

from hw_fakes import (  # noqa: F401
    _never_touch_real_hardware,
    clock,
    config,
    export,
    printer,
    roots,
    scanner,
    service,
)
