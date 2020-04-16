import torch


def video_transform(video, image_transform):
    """
    perform the image transformation and stack the video
    :param video: ndarray, with size [t, c, h, w]
    :param image_transform: the list of image transformation
    :return: video, tensor with size [c, t, h, w]
    """
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid
