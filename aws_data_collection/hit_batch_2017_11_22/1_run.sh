#!/usr/bin/env sh
#
# Copyright 2012 Amazon Technologies, Inc.
#
# Licensed under the Amazon Software License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
# http://aws.amazon.com/asl
#
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and
# limitations under the License.
export JAVA_HOME=/usr
SOURCE_FOLDER=$(pwd)

cd ../aws_command_line_tools/bin/

./loadHITs.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 -label $SOURCE_FOLDER/hit -input $SOURCE_FOLDER/hit.input -question $SOURCE_FOLDER/hit.question -properties $SOURCE_FOLDER/hit.properties


cd $SOURCE_FOLDER
