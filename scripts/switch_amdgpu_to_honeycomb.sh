#!/bin/sh
DEVICE_IDS=1002:1478,1002:1479,1002:73bf,1002:ab28,1002:73a6,1002:73a4

sudo modprobe -r amdgpu vfio_pci
echo 0000:41:00.0|sudo tee /sys/bus/pci/drivers/pcieport/unbind
echo 0000:42:00.0|sudo tee /sys/bus/pci/drivers/pcieport/unbind
#echo 0000:43:00.2|sudo tee /sys/bus/pci/drivers/xhci_hcd/unbind
#echo 0000:43:00.3|sudo tee /sys/bus/pci/drivers/i2c-designware-pci/unbind

sudo modprobe vfio_pci ids=$DEVICE_IDS disable_vga=1
sudo modprobe vfio_iommu_type1 allow_unsafe_interrupts=1
