# WIND via Optimal Estimation (WINDoe)

WINDoe is an optimal estimation algorithm to retrieve wind profiles from common wind profiling instrumentation such as Doppler lidar, radar wind profilers, and UAS. The advantage of WINDoe over traditional wind retrieval techniques is that information from the prior helps to constrain regions of the profile where observations have large uncertainty or the wind is under-determined from the observatins available. This allows for more information content from wind profiling instrumentation to be used in retrieving the wind profile, which increases the depths and eliminates vertical data gaps when compared to traditional methods. The design of WINDoe also makes it easy to retrieve wind profiles from using data from multiple instruments or to retrieve wind profiles from non-traditional Doppler lidar scans. WINDoe was designed in a very similar manner to TROPoe, which is a optimal estimation algorithm that retrieves thermodynamic profiles from ground based remote sensors. The retrieval is extremely flexible and the configuation is controlled by the variable-input parameter file (VIP). The variables in the VIP file will be described in a future User Guide.

## Dependencies
python 3.x

- os
- sys
- shutil
- copy
- numpy
- scipy
- netCDF4
- datetime
- calendar
- time
- argparse
- glob

Note: Most of these packages are standard in python distributions

## Instructions

It is planned that WINDoe will be put into a Docker container, but until then WINDoe can be run like so

```
python WINDoe.py /Path/to/vip/file /Path/to/prior/file/ [--shour <start hour>] [--ehour <end hour>] [--verbose <0,1,2,3>]
```

Monthly priors calculated from the radiosonde data at the ARM-SGP are included in the repository.  

## Disclamer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration (NOAA), or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis, with no warranty, and the user assumes responsibility for its use. NOAA has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information. Any claims against the Department of Commerce or NOAA stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## Authors

- [Dr. Joshua Gebauer](https://bliss.science/authors/joshua-gebauer/), NOAA National Severe Storms Laboratory / CIWRO, joshua.gebauer@noaa.gov

