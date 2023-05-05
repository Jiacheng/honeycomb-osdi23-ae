#!/bin/sh
DEVICE_IDS=1002:1478,1002:1479,1002:73bf,1002:ab28,1002:73a6,1002:73a4

sudo modprobe -r vfio_pci
sudo modprobe amdgpu
echo 0000:41:00.0|sudo tee /sys/bus/pci/drivers/pcieport/bind
echo 0000:42:00.0|sudo tee /sys/bus/pci/drivers/pcieport/bind
#echo 0000:43:00.2|sudo tee /sys/bus/pci/drivers/xhci_hcd/bind
#echo 0000:43:00.3|sudo tee /sys/bus/pci/drivers/i2c-designware-pci/bind