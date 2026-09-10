# Security and data handling

Do not commit microscope credentials, acquisition-computer hostnames, patient identifiers, protected health information, private specimen metadata, or vendor license keys.

Hardware adapters should default to simulation/dry-run behavior until explicit device configuration is supplied. Any future remote-control layer should include allowlisted commands, parameter bounds, timeouts, and emergency-stop behavior appropriate to the microscope platform.
