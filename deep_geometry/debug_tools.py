import matplotlib.pyplot as plt
import torch
import numpy as np


def de_normalize_image(images):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    return images * STD[:, None, None] + MEAN[:, None, None]  # de-normalize for vis


class DeepGeometrySetVisualizer:
    """
    Debug tools for deep geometry set training
    On k input images
    (1) visualize 20 key points on each image, show in one big pic
    (2) visualize 20 57*57 heatmap before normalization, show in one big pic
    (3) save 20 gaussian normalized heatmaps, show in one big pic
    (4) sum up 20 gaussian normalized heatmaps and save as one image.
    (4) save 20 vi-gaussian normalized heatmaps, show in one big pic
    (5) draw 20 key point connections as graphs on each image
    (6) draw 20 key point connections on gaussian normalized sum up heatmaps
    """

    def __init__(self, _save_dir, _name_prefix, _verbose=True):
        self.save_dir, self.verbose = _save_dir, _verbose
        self.name_prefix = _name_prefix

    def vis_debugger(self, batch_img_tensors, conv_heatmaps, gauss_heatmaps, vi_heatmaps, mu_xs, mu_ys,
                     show_graphs=False):
        self.vis_on_pure_background(mu_xs, mu_ys, show_graphs=show_graphs)
        self.vis_on_image(mu_xs, mu_ys, batch_img_tensors, show_graphs=show_graphs)
        self.vis_on_conv_heatmaps(mu_xs, mu_ys, conv_heatmaps, show_graphs=show_graphs)
        self.vis_on_gauss_heatmaps(mu_xs, mu_ys, gauss_heatmaps, show_graphs=show_graphs)
        print('vi heatmaps dim')
        print(vi_heatmaps.shape)
        self.vis_on_vi_heatmaps(mu_xs, mu_ys, vi_heatmaps, show_graphs=show_graphs)
        self.vis_heatmap_effects(mu_xs, mu_ys, conv_heatmaps, gauss_heatmaps, vi_heatmaps)

    def vis_on_pure_background(self, mu_xs, mu_ys, show_graphs=False):
        # draw a pure background and show points
        self.vis_points_on_map(mu_xs, mu_ys, None, save_file_prefix=self.name_prefix + '_pureback', de_normalize=False, show_graphs=show_graphs)
        return None

    def vis_on_image(self, mu_xs, mu_ys, batch_images, show_graphs=False):
        # visualize_key_points_and_graphs
        print(batch_images.shape)
        self.vis_points_on_map(mu_xs, mu_ys, batch_images.unsqueeze(1), save_file_prefix=self.name_prefix + '_image', de_normalize=True, show_graphs=show_graphs)
        return None

    def vis_on_conv_heatmaps(self, mu_xs, mu_ys, conv_heatmaps, show_graphs=False):
        # only draw on first heatmap
        self.vis_points_on_map(mu_xs, mu_ys, conv_heatmaps.unsqueeze(2), save_file_prefix=self.name_prefix + '_conv_heatmap', de_normalize=False,
                               show_graphs=show_graphs)
        return None

    def vis_on_gauss_heatmaps(self, mu_xs, mu_ys, gauss_heatmaps, show_graphs=False):
        # only draw on first heatmap
        self.vis_points_on_map(mu_xs, mu_ys, gauss_heatmaps.unsqueeze(2), save_file_prefix=self.name_prefix + '_gauss_heatmap', de_normalize=False,
                               show_graphs=show_graphs)
        return None

    def vis_on_vi_heatmaps(self, mu_xs, mu_ys, vi_heatmaps, show_graphs=False):
        # only draw on first heatmap
        self.vis_points_on_map(mu_xs, mu_ys, vi_heatmaps.unsqueeze(2), save_file_prefix=self.name_prefix + '_vi_heatmap', de_normalize=False,
                               show_graphs=show_graphs)
        return None

    def vis_points_on_map(self, mu_xs, mu_ys, heatmaps, save_file_prefix, de_normalize=False, show_graphs=False):
        """

        :param mu_xs:
        :param mu_ys:
        :param heatmaps: N*k*h*w if None, plot on pure background
        :param save_file_path:
        :param show_graphs:
        :return:
        """
        N, K = mu_xs.shape
        h, w = heatmaps.shape[3:5] if heatmaps is not None else (128, 128)
        if heatmaps is not None:
            print(save_file_prefix + 'heatmap dim')
            print(heatmaps.shape)
        print(save_file_prefix + ' image h {}, w{}'.format(h, w))
        # rectify the mu_xs and mu_ys
        mu_xs = (mu_xs + 1) * h / 2
        mu_ys = (mu_ys + 1) * w / 2
        for n in range(N):
            if heatmaps is not None:  # only plot on the first heatmap of k heatmaps
                img = heatmaps[n, 0] if de_normalize is False else de_normalize_image(heatmaps[n, 0])
                img.shape
                if img.shape[0]>1:  # rgb
                    plt.imshow(img.permute(1, 2, 0))
                else:  # single channel
                    plt.imshow(img[0], cmap='viridis')
            else:  # plot pure black color
                plt.imshow(np.zeros((128, 128, 3)))
            # plot the key points
            plt.scatter(x=mu_xs[n].tolist(), y=mu_ys[n].tolist(), c='r', s=20)

            if show_graphs:  # plot each graph
                l_color = 'b'

                def plot_line(idx1, idx2):
                    plt.plot([mu_xs[n, idx1].item(), mu_xs[n, idx2].item()],
                             [mu_ys[n, idx1].item(), mu_ys[n, idx2].item()],
                             color=l_color)

                # pp graphs
                plot_line(0, 1)
                plot_line(2, 3)
                plot_line(4, 5)
                # pl graphs
                plot_line(6, 7)
                plot_line(6, 8)
                plot_line(9, 10)
                plot_line(9, 11)
                # ll graphs
                plot_line(12, 14)
                plot_line(12, 15)
                plot_line(13, 14)
                plot_line(13, 15)

            plt.axis('off')
            # save
            plt.savefig(self.save_dir + '/' + save_file_prefix + '_N'+str(n) + '.png')
            plt.clf()

    def vis_heatmap_effects(self, mu_xs, mu_ys, conv_heatmaps, gauss_heatmaps, vi_heatmaps):
        # save 20 heatmaps on one figure for a batch of samples, plot mu_x, mu_y on each heatmap
        # save conv_heatmaps.png, gauss_heatmaps.png, vi_heatmaps.png
        # sum up 20 Gaussian heatmaps, show as one figure.
        self.vis_bulk_maps(mu_xs, mu_ys, conv_heatmaps, save_file_prefix=self.name_prefix + '_all_conv_heatmaps')
        self.vis_bulk_maps(mu_xs, mu_ys, gauss_heatmaps, save_file_prefix=self.name_prefix + '_all_gauss_heatmaps')
        self.vis_bulk_maps(mu_xs, mu_ys, vi_heatmaps, save_file_prefix=self.name_prefix + '_all_vi_heatmaps')
        # sum up 20 Gaussian heatmaps
        N, K = mu_xs.shape
        for n in range(N):
            sum_up_heatmaps = torch.sum(gauss_heatmaps[n], dim=0)
            plt.imshow(sum_up_heatmaps, cmap='viridis')
            plt.axis('off')
            # save
            plt.savefig(self.save_dir + '/'+self.name_prefix + '_gauss_sum_N' + str(n) + '.png')
            plt.clf()
        return None

    def vis_bulk_maps(self, mu_xs, mu_ys, heat_maps, save_file_prefix):
        N, K = mu_xs.shape
        h, w = heat_maps.shape[2:4] if heat_maps is not None else (128, 128)
        # rectify the mu_xs and mu_ys
        mu_xs = (mu_xs + 1) * h / 2
        mu_ys = (mu_ys + 1) * w / 2
        fig = plt.figure(figsize=(4, 5))
        columns = 4
        rows = 5
        for n in range(N):
            for k in range(1, columns * rows + 1):
                img = heat_maps[n, k-1]
                fig.add_subplot(rows, columns, k)
                plt.imshow(img, cmap='viridis')
                # plot the key point
                plt.scatter(x=mu_xs[n, k-1].item(), y=mu_ys[n, k-1].item(), c='r', s=3)
                plt.axis('off')
            # save
            plt.savefig(self.save_dir + '/' + save_file_prefix + '_N' + str(n) + '.png')
            plt.clf()
