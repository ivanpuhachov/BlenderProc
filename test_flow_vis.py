import h5py
import numpy as np
import matplotlib.pyplot as plt
# import cv2


def save_img_channels(
        x,
        save_as="renders/test.png",
        suptitle: str = None,
):
    plt.figure(figsize=(12, 4))
    n_channels = x.shape[-1]
    for i in range(x.shape[2]):
        plt.subplot(1, n_channels, i + 1)
        plt.imshow(x[..., i], cmap='coolwarm', interpolation="nearest")
        plt.colorbar()
        plt.title(f"[..., {i}]")
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.savefig(save_as, bbox_inches='tight', dpi=200)
    plt.close()


with h5py.File("examples/advanced/optical_flow/output/1.hdf5") as f:
    print(f.keys())
    im1 = np.array(f["colors"])
    print(im1.shape)
    ff = np.array(f["forward_flow"])
    print(ff.shape)
    save_img_channels(ff, "forward_flow.png")

with h5py.File("examples/advanced/optical_flow/output/0.hdf5") as f:
    print(f.keys())
    im2 = np.array(f["colors"])
    print(im2.shape)
    fb = np.array(f["backward_flow"])
    print(fb.shape)
    save_img_channels(ff, "backward_flow.png")

plt.figure()
plt.imshow(im1)
plt.show()


# h, w = flow.shape[:2]
# flow = -flow
# flow[:,:,0] += np.arange(w)
# flow[:,:,1] += np.arange(h)[:,np.newaxis]
# prevImg = cv2.remap(curImg, flow, None, cv.INTER_LINEAR)