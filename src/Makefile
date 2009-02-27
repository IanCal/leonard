################################################################################
#
#
#  This file is part of Leonard.
#
#  Foobar is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Foobar is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Leonard.  If not, see <http://www.gnu.org/licenses/>.
#
#
################################################################################
#
# Build script for project
#
################################################################################

# EDIT THIS LINE T
CUDA_SDK_INSTALL_PATH := /home/ian/NVIDIA_CUDA_SDK

# Add source files here
#EXECUTABLE	:=	leonard 
STATIC_LIB := leonard.so

CUFILES		:=  rbm.cu SimpleController.cu BasicFileInput.cu
CU_DEPS		:= 	kernels.cu rbm.cuh ParameterController.cuh SimpleController.cuh BasicFileInput.cu BasicFileInput.cuh
#CFILES		:=	example.cpp 

# Additional libraries needed by the project
USECUBLAS       := 1
USEALLEGRO		:= 1

################################################################################
# Rules and targets

include ../common.mk
