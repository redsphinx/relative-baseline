from relative_baseline.omg_emotion import visualization as VZ
from relative_baseline.omg_emotion.xai_tools import layer_visualization as layer_vis

import numpy as np
import torch


def add_kernels(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number in [2, 3, 71, 72, 73, 74, 75, 76, 77, 8, 11]:
        kernel = my_model.conv1.weight.data
        kernel = kernel.transpose(1, 2)

        for k in range(kernel.shape[0]):
            new_k = kernel[k].unsqueeze(0)

            project_variable.writer.add_video(tag='kernels/%d' % k, vid_tensor=new_k,
                                              global_step=project_variable.current_epoch, fps=2)
    elif model_number == 1:
        kernel = my_model.conv1.weight.data

        for k in range(kernel.shape[0]):
            new_k = kernel[k]
            project_variable.writer.add_image(tag='kernels/%d' % k, img_tensor=new_k,
                                              global_step=project_variable.current_epoch)


def add_temporal_visualizations(project_variable, my_model):
    def make_affine_matrix(ss, rr, xx, yy):
        matrix = np.zeros((2, 3))

        matrix[0, 0] = ss * np.cos(rr)
        matrix[0, 1] = -ss * np.sin(rr)
        matrix[0, 2] = xx * ss * np.cos(rr) - yy * ss * np.sin(rr)
        matrix[1, 0] = ss * np.sin(rr)
        matrix[1, 1] = ss * np.cos(rr)
        matrix[1, 2] = xx * ss * np.sin(rr) + yy * ss * np.cos(rr)

        return matrix
    
    model_number = project_variable.model_number

    if model_number in [11]:
        # conv 1
        scale = my_model.conv1.scale.data
        rotate = my_model.conv1.rotate.data
        translate_x = my_model.conv1.translate_x.data
        translate_y = my_model.conv1.translate_y.data

        # for each channel
        for i in range(scale.shape[1]):
            pacman_frames = np.zeros((scale.shape[0]+1, 100, 100))
            pacman_frames[0] = VZ.load_og_pacman()

        # for each time-step
            for j in range(scale.shape[0]):
        # take SRXY and transform into affine matrix
                s = float(scale[j, i].cpu())
                r = float(rotate[j, i].cpu())
                x = float(translate_x[j, i].cpu())
                y = float(translate_y[j, i].cpu())
                
                affine_matrix = make_affine_matrix(s, r, x, y)
                pacman_frames[j+1] = VZ.make_pacman_frame(pacman_frames[j], affine_matrix)

            # save img as video
            pacman_frames = np.expand_dims(pacman_frames, axis=0)
            pacman_frames = np.expand_dims(pacman_frames, axis=2)
            project_variable.writer.add_video(tag='channel/%d' % i, vid_tensor=pacman_frames,
                                          global_step=project_variable.current_epoch, fps=10)



def add_histograms(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number in [1, 2, 3, 71, 72, 73, 74, 75, 76, 77]:
        project_variable.writer.add_histogram('fc1/weight', my_model.fc1.weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc2/weight', my_model.fc2.weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc3/weight', my_model.fc3.weight, project_variable.current_epoch)

        project_variable.writer.add_histogram('fc1/bias', my_model.fc1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc2/bias', my_model.fc2.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc3/bias', my_model.fc3.bias, project_variable.current_epoch)


    if model_number == 3:
        project_variable.writer.add_histogram('first_weight/conv1', my_model.conv1.first_weight,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('first_weight/conv2', my_model.conv2.first_weight,
                                              project_variable.current_epoch)

        if project_variable.theta_init is None:
            project_variable.writer.add_histogram('conv1.weight/scale', my_model.conv1.scale,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/rotate', my_model.conv1.rotate,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/translate_x', my_model.conv1.translate_x,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/translate_y', my_model.conv1.translate_y,
                                                  project_variable.current_epoch)

            project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)

            project_variable.writer.add_histogram('conv2.weight/scale', my_model.conv2.scale,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.weight/rotate', my_model.conv2.rotate,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.weight/translate_x', my_model.conv2.translate_x,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.weight/translate_y', my_model.conv2.translate_y,
                                                  project_variable.current_epoch)

            project_variable.writer.add_histogram('conv2.bias', my_model.conv2.bias, project_variable.current_epoch)
        else:
            project_variable.writer.add_histogram('conv1.theta', my_model.conv1.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.theta', my_model.conv2.theta, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.bias', my_model.conv2.bias, project_variable.current_epoch)

    elif model_number in [71, 72, 73, 75, 76, 77]:
        if project_variable.theta_init is None:
            # TODO: implement
            pass
        else:
            project_variable.writer.add_histogram('conv1.theta', my_model.conv1.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.theta', my_model.conv2.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv3.theta', my_model.conv3.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv4.theta', my_model.conv4.theta, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.bias', my_model.conv2.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv3.bias', my_model.conv3.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv4.bias', my_model.conv4.bias, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.k0', my_model.conv1.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.k0', my_model.conv2.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv3.k0', my_model.conv3.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv4.k0', my_model.conv4.first_weight, project_variable.current_epoch)


    elif model_number in [74]:
        if project_variable.theta_init is None:
            # TODO: implement
            pass
        else:
            project_variable.writer.add_histogram('conv1.theta', my_model.conv1.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv2.theta', my_model.conv2.theta, project_variable.current_epoch)
            project_variable.writer.add_histogram('conv3.theta', my_model.conv3.theta, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.bias', my_model.conv2.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv3.bias', my_model.conv3.bias, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.k0', my_model.conv1.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.k0', my_model.conv2.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv3.k0', my_model.conv3.first_weight, project_variable.current_epoch)

    elif model_number in [8]:
        if project_variable.theta_init is None:
            project_variable.writer.add_histogram('conv1.weight/scale', my_model.conv1.scale,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/rotate', my_model.conv1.rotate,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/translate_x', my_model.conv1.translate_x,
                                                  project_variable.current_epoch)
            project_variable.writer.add_histogram('conv1.weight/translate_y', my_model.conv1.translate_y,
                                                  project_variable.current_epoch)
        else:
            project_variable.writer.add_histogram('conv1.theta', my_model.conv1.theta, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.k0', my_model.conv1.first_weight, project_variable.current_epoch)



def add_scalars(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number in [3, 8]:
        if project_variable.theta_init is None:
            s = my_model.conv1.scale.data
            r = my_model.conv1.rotate.data
            x = my_model.conv1.translate_x.data
            y = my_model.conv1.translate_y.data

            if s.shape[0] == 1:
                for i in range(s.shape[1]):
                    project_variable.writer.add_scalar('k%d/scale' % i, s[0, i], project_variable.current_epoch)
                    project_variable.writer.add_scalar('k%d/rotate' % i, r[0, i], project_variable.current_epoch)
                    project_variable.writer.add_scalar('k%d/translate_x' % i, x[0, i], project_variable.current_epoch)
                    project_variable.writer.add_scalar('k%d/translate_y' % i, y[0, i], project_variable.current_epoch)
            else:
                for i in range(s.shape[1]):
                    project_variable.writer.add_scalars('k%d/scale' % i,
                                                        {"t0": s[0, i], "t1": s[1, i], "t2": s[2, i], "t3": s[3, i]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/rotate' % i,
                                                        {"t0": r[0, i], "t1": r[1, i], "t2": r[2, i], "t3": r[3, i]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/translate_x' % i,
                                                        {"t0": x[0, i], "t1": x[1, i], "t2": x[2, i], "t3": x[3, i]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/translate_y' % i,
                                                        {"t0": y[0, i], "t1": y[1, i], "t2": y[2, i], "t3": y[3, i]},
                                                        project_variable.current_epoch)

        else:
            theta = my_model.conv1.theta.data
            for i in range(theta.shape[1]):
                for j in range(theta.shape[0]):
                    project_variable.writer.add_scalars('k%d/theta[0, 0]' % i, {"t%d" % j: theta[j, i, 0, 0]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/theta[0, 1]' % i, {"t%d" % j: theta[j, i, 0, 1]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/theta[0, 2]' % i, {"t%d" % j: theta[j, i, 0, 2]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/theta[1, 0]' % i, {"t%d" % j: theta[j, i, 1, 0]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/theta[1, 1]' % i, {"t%d" % j: theta[j, i, 1, 1]},
                                                        project_variable.current_epoch)
                    project_variable.writer.add_scalars('k%d/theta[1, 2]' % i, {"t%d" % j: theta[j, i, 1, 2]},
                                                        project_variable.current_epoch)



def add_standard_info(project_variable, which, parameters):
    loss, accuracy, confusion_epoch = parameters

    project_variable.writer.add_scalar('loss/%s' % which, loss, project_variable.current_epoch)
    project_variable.writer.add_scalar('accuracy/%s' % which, accuracy, project_variable.current_epoch)
    fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    project_variable.writer.add_figure(tag='confusion/%s' % which, figure=fig, global_step=project_variable.current_epoch)


def add_xai(project_variable, my_model, device, data_point=None):
    # assert(project_variable.model_number in [11, 12])

    if 'erhan2009' in project_variable.which_methods:
        all_outputs = layer_vis.run_erhan2009(project_variable, my_model, device)

        which_methods = 'erhan2009'

        for j in range(len(project_variable.which_layers)):
            for k in range(len(project_variable.which_channels[j])):
                which_layers = project_variable.which_layers[j]
                which_channels = project_variable.which_channels[j][k]

                output = all_outputs[j][k]

                # output = np.expand_dims(np.expand_dims(output, axis=0), axis=0)
                output = output.transpose(0, 2, 1, 3, 4)

                # example: xai/erhan2009/layer1/channel
                project_variable.writer.add_video(tag='xai/%s/%s/channel %d' % (which_methods, which_layers,
                                                                                which_channels),
                                                  vid_tensor=output,
                                                  global_step=project_variable.current_epoch, fps=5)

    if 'zeiler2014' in project_variable.which_methods:
        assert(data_point is not None)

        all_outputs = layer_vis.run_zeiler2014(project_variable, data_point, my_model, device)
        which_methods = 'zeiler2014'

        for j in range(len(project_variable.which_layers)):
            for k in range(len(project_variable.which_channels[j]+1)):
                which_layers = project_variable.which_layers[j]
                which_channels = project_variable.which_channels[j][k]

                output = all_outputs[j][k]

                # output = np.expand_dims(np.expand_dims(output, axis=0), axis=0)
                output = output.transpose(0, 2, 1, 3, 4)

                # example: xai/erhan2009/layer1/channel
                project_variable.writer.add_video(tag='xai/%s/%s/channel %d' % (which_methods, which_layers,
                                                                                which_channels),
                                                  vid_tensor=output,
                                                  global_step=project_variable.current_epoch, fps=5)

    if 'gradient_method' in project_variable.which_methods:
        assert (data_point is not None)

        # TODO add mode for resnet18's

        # mode = 'slices'
        if project_variable.model_number in [11, 14, 15, 16, 20]:
            mode = 'srxy' # only for 3DTTN
        else:
            mode = None

        which_methods = 'gradient_method'
        dp, rest, optional_srxy = layer_vis.gradient_method(project_variable, data_point, my_model, device, mode)

        if mode == 'srxy':
            assert optional_srxy is not None

            for j in range(len(project_variable.which_layers)):
                for k in range(len(project_variable.which_channels[j] + 1)):
                    which_layers = project_variable.which_layers[j]
                    which_channels = project_variable.which_channels[j][k]

                    temporal_dim = len(rest[0][0])
                    if project_variable.dataset == 'jester':
                        _, c_, h_, w_ = rest[0][0][0].shape
                    else:
                        _, h_, w_ = rest[0][0][0].shape
                        c_ = 1

                    output = np.zeros(shape=(temporal_dim + 1, c_, h_, w_), dtype=np.uint8)
                    output[0] = dp

                    for t in range(temporal_dim):
                        output[t + 1] = rest[j][:][k][t]

                    output = np.expand_dims(output, 0)

                    project_variable.writer.add_video(tag='xai/%s/%s/channel %d' % (which_methods, which_layers,
                                                                                    which_channels),
                                                      vid_tensor=output,
                                                      global_step=project_variable.current_epoch, fps=2)

                    fig = VZ.plot_srxy(optional_srxy, j, k)
                    project_variable.writer.add_figure(tag='srxy_params/layer_%d/channel_%d'
                                                           % (j+1, k+1), figure=fig,
                                                       global_step=project_variable.current_epoch)

            project_variable.writer.add_image(tag='xai/%s/0_original' % (which_methods),
                                              img_tensor=dp,
                                              global_step=project_variable.current_epoch)

        else:
            for j in range(len(project_variable.which_layers)):
                for k in range(len(project_variable.which_channels[j])):
                    which_layers = project_variable.which_layers[j]
                    which_channels = project_variable.which_channels[j][k]

                    temporal_dim = len(rest[0][0])
                    _, h_, w_ = rest[0][0][0].shape

                    output = np.zeros(shape=(temporal_dim, 1, h_, w_), dtype=np.uint8)

                    for t in range(temporal_dim):
                        output[t] = rest[j][k][t]

                    output = np.expand_dims(output, 0)

                    project_variable.writer.add_video(tag='xai/%s/%s/channel %d' % (which_methods, which_layers,
                                                                                    which_channels),
                                                      vid_tensor=output,
                                                      global_step=project_variable.current_epoch, fps=2)

            dp = np.array(dp)
            dp = np.expand_dims(dp, 0)
            project_variable.writer.add_video(tag='xai/%s/0_original' % (which_methods),
                                              vid_tensor=dp,
                                              global_step=project_variable.current_epoch, fps=2)


def add_histograms_srxy(project_variable, my_model):
    if project_variable.model_number == 11:

        project_variable.writer.add_histogram('conv1.weight/scale', my_model.conv1.scale,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/rotate', my_model.conv1.rotate,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/translate_x', my_model.conv1.translate_x,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/translate_y', my_model.conv1.translate_y,
                                              project_variable.current_epoch)

        project_variable.writer.add_histogram('conv2.weight/scale', my_model.conv2.scale,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/rotate', my_model.conv2.rotate,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/translate_x', my_model.conv2.translate_x,
                                              project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/translate_y', my_model.conv2.translate_y,
                                              project_variable.current_epoch)
    else:
        print('model number = %d, no srxy histograms plotted' % (project_variable.model_number))


def add_text_srxy_per_channel(project_variable, my_model):
    if project_variable.model_number == 11:
        # conv1

        conv1_scale = my_model.conv1.scale
        conv1_rotate = my_model.conv1.rotate
        conv1_translate_x = my_model.conv1.translate_x
        conv1_translate_y = my_model.conv1.translate_y

        conv1_string = ''

        tr, ch = conv1_scale.shape

        for i in range(ch):
            for j in range(tr):
                tmp_ = 'channel %d, trnsf %d, scale %0.2f, rotate %0.2f, trnsl x %0.2f, trnsl y %0.2f  \n' \
                       % (i, j, conv1_scale[j, i], conv1_rotate[j, i], conv1_translate_x[j, i], conv1_translate_y[j, i])
                conv1_string = conv1_string + tmp_

        project_variable.writer.add_text('conv1/', conv1_string)

        # conv2
        conv2_scale = my_model.conv2.scale
        conv2_rotate = my_model.conv2.rotate
        conv2_translate_x = my_model.conv2.translate_x
        conv2_translate_y = my_model.conv2.translate_y

        conv2_string = ''

        tr, ch = conv2_scale.shape

        for i in range(ch):
            for j in range(tr):
                tmp_ = 'channel %d, trnsf %d, scale %0.2f, rotate %0.2f, trnsl x %0.2f, trnsl y %0.2f  \n' \
                       % (i, j, conv2_scale[j, i], conv2_rotate[j, i], conv2_translate_x[j, i], conv2_translate_y[j, i])
                conv2_string = conv2_string + tmp_

        project_variable.writer.add_text('conv2/', conv2_string)
    else:
        print('model number = %d, no srxy histograms plotted' % (project_variable.model_number))