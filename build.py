import os

import torch
from torch.utils.ffi import create_extension

sources = ['nms_c.c']
headers = ['nms_c.h']
defines = []
defines += [('WITH_CUDA', None)]
with_cuda = True
extra_objects = ['nms.cu.o']
this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(extra_objects)

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
