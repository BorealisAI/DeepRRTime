# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see https://github.com/salesforce/DeepTime/blob/main/LICENSE.txt

for dataset in ECL ETTm2 Exchange ILI Traffic Weather; do
  for instance in `/bin/ls -d storage/experiments/$dataset/*/*`; do
      echo $instance
      make run command=${instance}/command
  done
done
