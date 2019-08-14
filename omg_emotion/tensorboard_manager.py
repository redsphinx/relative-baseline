from relative_baseline.omg_emotion import visualization as VZ


def add_kernels(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number in [2, 3, 7]:
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

    else:
        print('ERROR: model number %d not supported. Kernels will not be added to TensorBoard.' % model_number)


def add_histograms(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number in [1, 2, 3, 7]:
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

    elif model_number == 7:
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




def add_scalars(project_variable, my_model):
    model_number = project_variable.model_number

    if model_number == [3]:
        if project_variable.theta_init is None:
            s = my_model.conv1.scale.data
            r = my_model.conv1.rotate.data
            x = my_model.conv1.translate_x.data
            y = my_model.conv1.translate_y.data

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
            theta = my_model.theta.data
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
