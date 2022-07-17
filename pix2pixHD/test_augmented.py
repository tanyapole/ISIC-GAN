### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torchvision.transforms.functional as TFF
from pathlib import Path


def get_augmented(img, idx):
    img = img.clone()
    if idx == 0:
        return img
    elif idx == 1:
        return TFF.hflip(img)
    elif idx == 2:
        return TFF.vflip(img)
    elif idx == 3:
        return TFF.vflip(TFF.hflip(img))
    else:
        raise Exception(f"Unsupported augmentation idx={idx}")


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)

    # augment
    label, inst = data['label'], data['inst']
    label, inst = label.squeeze(0).squeeze(0), inst.squeeze(0).squeeze(0)
    for j in range(4):
        label2, inst2 = get_augmented(label, j), get_augmented(inst, j)
        label2, inst2 = label2.unsqueeze(0).unsqueeze(0), inst2.unsqueeze(0).unsqueeze(0)

        minibatch = 1 
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [label2, inst2])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [label2, inst2])
        else:        
            generated = model.inference(label2, inst2, data['image'])
            
        visuals = OrderedDict([('input_label', util.tensor2label(label2[0], opt.label_nc)),
                            ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path'][0]
        _pt = Path(img_path)
        img_path = str(_pt.parent / (_pt.name[:-len(_pt.suffix)] + f'_v{j}' + _pt.suffix))
        img_path = [img_path]
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

webpage.save()
