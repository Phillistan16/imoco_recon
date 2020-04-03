import argparse
import sigpy as sp
import numpy as np
from sigpy_e import cfl, ext
from sigpy_e.linop_e import NFTs, Diags
import time
import cupy as cp

## IO parameters
parser = argparse.ArgumentParser(description="XD-GRASP recon.")

parser.add_argument(
    "--res_scale", type=float, default=0.75, help="scale of resolution, full res == .75"
)
# Would be good to implement image support version of autoFOV based on espirit maps (nishimura, I think)
parser.add_argument("--fov_x", type=float, default=1, help="scale of FOV x, full res == 1")
parser.add_argument("--fov_y", type=float, default=1, help="scale of FOV y, full res == 1")
parser.add_argument("--fov_z", type=float, default=1, help="scale of FOV z, full res == 1")

parser.add_argument(
    "--lambda_TV", type=float, default=5e-2, help="TV regularization, default is 0.05"
)
parser.add_argument("--outer_iter", type=int, default=20, help="Num of Iterations.")

parser.add_argument("--device", type=int, default=0, help="Computing device.")

parser.add_argument("fname", type=str, help="Prefix of raw data and output(_mrL).")
args = parser.parse_args()
xp = sp.Device(args.device).xp
#
res_scale = args.res_scale
fname = args.fname
lambda_TV = args.lambda_TV
device = args.device
outer_iter = args.outer_iter
fov_scale = (args.fov_x, args.fov_y, args.fov_z)

## data loading
data = cfl.read_cfl(fname + "_datam")
traj = np.real(cfl.read_cfl(fname + "_trajm"))
dcf = cfl.read_cfl(fname + "_dcf2m")
nf_scale = res_scale
nf_arr = np.sqrt(np.sum(traj[0, 0, 0, 0, :, :] ** 2, axis=1))
nf_e = np.sum(nf_arr < np.max(nf_arr) * nf_scale)

scale = fov_scale
traj[..., 0] = traj[..., 0] * scale[0]
traj[..., 1] = traj[..., 1] * scale[1]
traj[..., 2] = traj[..., 2] * scale[2]

traj = traj[..., :nf_e, :]
data = data[..., :nf_e, :]
dcf = dcf[..., :nf_e, :]

print("Image Shape Estimate: {}".format(sp.estimate_shape(traj)))
nphase, nEcalib, nCoil, npe, nfe, _ = data.shape
tshape = (
    np.int(np.max(traj[..., 0]) - np.min(traj[..., 0])),
    np.int(np.max(traj[..., 1]) - np.min(traj[..., 1])),
    np.int(np.max(traj[..., 2]) - np.min(traj[..., 2])),
)

### calibration
print("running calibration")
ksp = np.reshape(np.transpose(data, (2, 1, 0, 3, 4, 5)), (nCoil, nphase * npe, nfe))
dcf2 = np.reshape(np.transpose(dcf, (2, 1, 0, 3, 4, 5)), (nphase * npe, nfe))
coord = np.reshape(np.transpose(traj, (2, 1, 0, 3, 4, 5)), (nphase * npe, nfe, 3))

mps = ext.jsens_calib(ksp, coord, dcf2, device=sp.Device(0), ishape=tshape)
S = sp.linop.Multiply(tshape, sp.to_device(mps, device))

del ksp, dcf2, coord, mps
### recon
print("starting recon")
# Generate Linop. If we disect this we can fit into GPU memory.
PFTSs = []
for i in range(nphase):
    FTs = NFTs((nCoil,) + tshape, sp.to_device(traj[i, 0, 0, ...], device), device)
    W = sp.linop.Multiply((nCoil, npe, nfe,), sp.to_device(dcf[i, 0, 0, :, :, 0], device))
    FTSs = W * FTs * S
    PFTSs.append(FTSs)
PFTSs = Diags(PFTSs, oshape=(nphase, nCoil, npe, nfe,), ishape=(nphase,) + tshape)
del FTs, W, FTSs

cp._default_memory_pool.free_all_blocks()
cp._default_pinned_memory_pool.free_all_blocks()


## preconditioner
print("running preconditioner")
timeI = time.time()
wdata = data[:, 0, :, :, :, 0] * dcf[:, 0, :, :, :, 0]
L = xp.mean(xp.abs(PFTSs.H * PFTSs * xp.ones((nphase,) + tshape, dtype="complex64")))
timeF = time.time()
print("Time for preconditioner: {} seconds.".format(timeF - timeI))

## reconstruction
print("running reconstruction")
q2 = xp.zeros((nphase,) + tshape, dtype=xp.complex64)
Y = np.zeros_like(wdata)
q20 = xp.zeros_like(q2)

sigma = 0.4
tau = 0.4
for i in range(outer_iter):
    timeI = time.time()
    q2 = sp.to_device(q2, device)
    q20 = sp.to_device(1 / L * (PFTSs * q2), -1)
    Y = (Y + sigma * (q20 - wdata)) / (1 + sigma)
    q20 = sp.to_device(q2, -1)
    q2 = ext.TVt_prox(
        sp.to_device(q2, -1) - tau * sp.to_device(PFTSs.H * sp.to_device(Y, device), -1), lambda_TV
    )
    timeF = time.time()
    print(
        "outer iter:{}, res:{}, time:{} seconds".format(
            i, np.linalg.norm(q2 - q20) / np.linalg.norm(q2), timeF - timeI
        )
    )
print("writing data...")
cfl.write_cfl(fname + "_mrL", q2)
print("done...")
