import torch.nn as nn
import torch
import math
import numpy as np
import copy


class NSCTdec:
    def __init__(self, levels, device):
        self.device = device
        self.levels = levels
        self.max_filter = nn.MaxPool2d((9, 9), stride=(1, 1), padding=(4, 4))
        self.dKernel = torch.ones(1, 1, 3, 3).to(self.device) / 9
        self.stdpadding = nn.ReflectionPad2d((1, 1, 1, 1))
        filters = {}
        h1, h2 = self.dfilters()
        filters["0"] = (
            torch.from_numpy(self.modulate2(h1, "c")[None, None, :, :])
            .float()
            .to(self.device)
        )
        filters["1"] = (
            torch.from_numpy(self.modulate2(h2, "c")[None, None, :, :])
            .float()
            .to(self.device)
        )
        filters["2"], filters["3"] = self.parafilters(h1, h2)
        self.h1, self.h2 = self.atrousfilters()
        self.filters = filters

    def atrousfilters(self):
        A = np.array(
            [
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
                [-0.01294417, 0.0625, 0.15088835, 0.0625, -0.01294417],
                [-0.01941626, 0.15088835, 0.34060922, 0.15088835, -0.01941626],
                [-0.01294417, 0.0625, 0.15088835, 0.0625, -0.01294417],
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
            ]
        )
        B = np.array(
            [
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
                [-0.01294417, -0.0625, -0.09911165, -0.0625, -0.01294417],
                [-0.01941626, -0.09911165, 0.84060922, -0.09911165, -0.01941626],
                [-0.01294417, -0.0625, -0.09911165, -0.0625, -0.01294417],
                [-0.00323604, -0.01294417, -0.01941626, -0.01294417, -0.00323604],
            ]
        )
        return torch.from_numpy(A)[None, None, :, :].float().to(
            self.device
        ), torch.from_numpy(B)[None, None, :, :].float().to(self.device)

    def parafilters(self, f1, f2):
        y1, y2 = {}, {}
        y1["0"], y2["0"] = self.modulate2(f1, "r"), self.modulate2(f2, "r")
        y1["1"], y2["1"] = self.modulate2(f1, "c"), self.modulate2(f2, "c")
        y1["2"], y2["2"] = y1["0"].T, y2["0"].T
        y1["3"], y2["3"] = y1["1"].T, y2["1"].T
        for i in range(4):
            y1["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y1["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
            y2["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y2["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
        return y1, y2

    def resampz(self, x, sampleType):
        sx = x.shape
        # TODO: check the code below
        # shift, sx = 1, x.shape
        if (sampleType == 1) or (sampleType == 2):
            y = np.zeros((sx[0] + sx[1] - 1, sx[1]))
            shift1 = (
                -1 * np.arange(0, sx[1], 1, dtype=int)
                if sampleType == 1
                else np.arange(0, sx[1], 1, dtype=int)
            )
            if shift1[-1] < 0:
                shift1 = shift1 - shift1[-1]
            for n in range(sx[1]):
                y[shift1[n] + np.arange(0, sx[0], 1, dtype=int), n] = x[:, n]
            start, finish = 0, y.shape[0] - 1
            while np.sum(np.abs(y[start, :])) == 0:
                start = start + 1
            while np.sum(np.abs(y[finish, :])) == 0:
                finish = finish - 1
            y = y[start : finish + 1, :]
        else:
            y = np.zeros((sx[0], sx[1] + sx[0] - 1))
            shift2 = (
                -1 * np.arange(0, sx[0], 1, dtype=int)
                if sampleType == 3
                else np.arange(0, sx[0], 1, dtype=int)
            )
            if shift2[-1] < 0:
                shift2 = shift2 - shift2[-1]
            for m in range(sx[0]):
                y[m, shift2[m] + np.arange(0, sx[1], 1, dtype=int)] = x[m, :]
            start, finish = 0, y.shape[1] - 1
            while np.sum(np.abs(y[:, start])) == 0:
                start = start + 1
            while np.sum(np.abs(y[:, finish])) == 0:
                finish = finish - 1
            y = y[:, start : finish + 1]
        return y

    def modulate2(self, x, modulateType):
        o = np.floor(np.array(x.shape) / 2) + 1
        n1, n2 = (
            np.arange(1, x.shape[0] + 1, 1) - o[0],
            np.arange(1, x.shape[1] + 1, 1) - o[1],
        )
        if modulateType == "c":
            m2 = (-1) ** n2
            return x * np.repeat(m2[None, :], x.shape[0], axis=0)
        elif modulateType == "r":
            m1 = (-1) ** n1
            return x * np.repeat(m1[:, None], x.shape[1], axis=1)

    def dfilters(self):
        A = np.array([[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]])
        B = np.array(
            [
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
                [-0.0, -0.125, -0.25, -0.125, -0.0],
                [-0.0625, -0.25, 1.75, -0.25, -0.0625],
                [-0.0, -0.125, -0.25, -0.125, -0.0],
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
            ]
        )
        return A / math.sqrt(2), B / math.sqrt(2)

    def nsfbdec(self, x, h0, h1, lev):
        if lev != 0:
            y0 = torch.conv2d(
                self.symext(
                    x,
                    (2 ** (lev - 1)) * (h0.size(-2) - 1),
                    (2 ** (lev - 1)) * (h0.size(-1) - 1),
                ),
                h0,
                dilation=2**lev,
            )
            y1 = torch.conv2d(
                self.symext(
                    x,
                    (2 ** (lev - 1)) * (h1.size(-2) - 1),
                    (2 ** (lev - 1)) * (h1.size(-1) - 1),
                ),
                h1,
                dilation=2**lev,
            )
        else:
            y0, y1 = torch.conv2d(
                self.symext(x, h0.size(-2) // 2, h0.size(-1) // 2), h0
            ), torch.conv2d(self.symext(x, h1.size(-2) // 2, h1.size(-1) // 2), h1)
        return y0, y1

    def symext(self, x, er, ec):
        x = torch.cat(
            (torch.flip(x[:, :, :er, :], [-2]), x, torch.flip(x[:, :, -er:, :], [-2])),
            -2,
        )
        return torch.cat(
            (torch.flip(x[:, :, :, :ec], [-1]), x, torch.flip(x[:, :, :, -ec:], [-1])),
            -1,
        )

    def nsdfbdec(self, x, dfilter, clevels):
        k1, k2, f1, f2 = dfilter["0"], dfilter["1"], dfilter["2"], dfilter["3"]
        q1 = np.array([[1, -1], [1, 1]])
        if clevels == 1:
            y = self.nssfbdec(x, k1, k2)
        else:
            tmp = self.nssfbdec(x, k1, k2)
            x1, x2 = tmp[:, 0:1, :, :], tmp[:, 1:2, :, :]
            y = torch.cat(
                (self.nssfbdec(x1, k1, k2, q1), self.nssfbdec(x2, k1, k2, q1)), 1
            )
            for ll in range(3, clevels + 1):
                y_old = y
                y = torch.zeros(
                    x.size()[0], 2**ll, y_old.size()[2], y_old.size()[3]
                ).to(self.device)
                for k in range(1, 2 ** (ll - 2) + 1):
                    slk = 2 * math.floor((k - 1) / 2) - 2 ** (ll - 3) + 1
                    mkl = 2 * np.matmul(
                        np.array([[2 ** (ll - 3), 0], [0, 1]]),
                        np.array([[1, 0], [-slk, 1]]),
                    )
                    i = (k - 1) % 2 + 1
                    y[:, 2 * k - 2 : 2 * k, :, :] = self.nssfbdec(
                        y_old[:, k - 1 : k, :, :],
                        f1["{}".format(i - 1)],
                        f2["{}".format(i - 1)],
                        mkl,
                    )
                for k in range(2 ** (ll - 2) + 1, 2 ** (ll - 1) + 1):
                    slk = (
                        2 * math.floor((k - 2 ** (ll - 2) - 1) / 2) - 2 ** (ll - 3) + 1
                    )
                    mkl = 2 * np.matmul(
                        np.array([[1, 0], [0, 2 ** (ll - 3)]]),
                        np.array([[1, -slk], [0, 1]]),
                    )
                    i = (k - 1) % 2 + 3
                    y[:, 2 * k - 2 : 2 * k, :, :] = self.nssfbdec(
                        y_old[:, k - 1 : k, :, :],
                        f1["{}".format(i - 1)],
                        f2["{}".format(i - 1)],
                        mkl,
                    )
        return y

    def nssfbdec(self, x, f1, f2, mup="None"):
        if isinstance(mup, str):
            return torch.cat((self.efilter2(x, f1), self.efilter2(x, f2)), 1)
        if isinstance(mup, int):
            if mup == 1:
                return torch.cat((self.efilter2(x, f1), self.efilter2(x, f2)), 1)
        if sum(sum(mup == np.eye(2))) == 4:
            return torch.cat((self.efilter2(x, f1), self.efilter2(x, f2)), 1)
        if mup.shape == (2, 2):
            return torch.cat((self.zconv2(x, f1, mup), self.zconv2(x, f2, mup)), 1)

    def zconv2(self, x, h, m):
        h0 = (
            self.rot45(h)
            if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
            else self.my_upsamp2df(h, m)
        )
        return torch.conv2d(self.perext(x, h0.size(-2) // 2, h0.size(-1) // 2), h0)

    def efilter2(self, x, f):
        return torch.conv2d(self.perext(x, f.size(-2) // 2, f.size(-1) // 2), f)

    def rot45(self, h0):
        h = torch.zeros(1, 1, 2 * h0.size()[-2] - 1, 2 * h0.size()[-1] - 1).to(
            self.device
        )
        sz1, sz2 = h0.size()[-2:]
        for i in range(1, sz1 + 1):
            r, c = i + np.arange(0, sz2, 1), sz2 - i + np.arange(1, sz2 + 1, 1)
            for j in range(1, sz2 + 1):
                h[:, :, r[j - 1] - 1, c[j - 1] - 1] = h0[:, :, i - 1, j - 1]
        return h

    def my_upsamp2df(self, h0, mup):
        m, n = h0.size()[-2:]
        power = math.log2(mup[0, 0])
        R1, R2 = torch.zeros((1, 1, int(2**power * (m - 1) + 1), m)).to(
            self.device
        ), torch.zeros((1, 1, n, int(2**power * (n - 1) + 1))).to(self.device)
        for i in range(1, m + 1):
            R1[:, :, int((i - 1) * 2 ** (power)), i - 1] = 1
        for i in range(1, n + 1):
            R2[:, :, i - 1, int((i - 1) * 2 ** (power))] = 1
        return torch.matmul(torch.matmul(R1, h0), R2)

    def perext(self, x, er, ec):
        x = torch.cat((x[:, :, -er:, :], x, x[:, :, :er, :]), -2)
        return torch.cat((x[:, :, :, -ec:], x, x[:, :, :, :ec]), -1)

    def extractFeatures(self, x):
        b, _, m, n = x[0].size()
        f = torch.zeros(b, 1, m, n).to(self.device)
        L = sum([2**ll for ll in self.levels])
        for d in x:
            f += torch.sum(d.abs(), dim=1, keepdim=True)
        return f / L

    def nsctDec(self, x, stride=None, _forFeatures=False):
        clevels, nIndex = len(self.levels), len(self.levels) + 1
        y = []
        for i in range(1, clevels + 1):
            xlo, xhi = self.nsfbdec(x, self.h1, self.h2, i - 1)
            if self.levels[nIndex - 2] > 0:
                xhi_dir = self.nsdfbdec(xhi, self.filters, self.levels[nIndex - 2])
                y.append(xhi_dir)
            else:
                y.append(xhi)
            nIndex = nIndex - 1
            x = xlo
        if _forFeatures:
            f = self.extractFeatures(y)
            df, dfbase = torch.conv2d(
                f, self.dKernel, stride=stride, padding=self.dKernel.shape[-1] // 2
            ), torch.conv2d(
                x, self.dKernel, stride=stride, padding=self.dKernel.shape[-1] // 2
            )
            dfstd = (
                self.stdpadding(df).unfold(2, 3, 1).unfold(3, 3, 1).std(dim=(-2, -1))
            )
            del f, x
            return df[:, 0, :, :], dfbase[:, 0, :, :], dfstd[:, 0, :, :]
        else:
            return y, x


class NSCTrec:
    def __init__(self, levels, device):
        self.device = device
        self.levels = levels
        h1, h2 = self.dfilters()
        filters = {}
        filters["0"] = (
            torch.from_numpy(self.modulate2(h1, "c")[None, None, :, :])
            .float()
            .to(self.device)
        )
        filters["1"] = (
            torch.from_numpy(self.modulate2(h2, "c")[None, None, :, :])
            .float()
            .to(self.device)
        )
        filters["2"], filters["3"] = self.parafilters(h1, h2)
        self.g1, self.g2 = self.atrousfilters()
        self.filters = filters

    def nssfbrec(self, x1, x2, f1, f2, mup="None"):
        if isinstance(mup, str):
            y1, y2 = self.efilter2(x1, f1), self.efilter2(x2, f2)
            return y1 + y2
        if sum(sum(mup == np.eye(2))) == 4:
            y1, y2 = self.efilter2(x1, f1), self.efilter2(x2, f2)
            return y1 + y2
        if isinstance(mup, int):
            if mup == 1:
                y1, y2 = self.efilter2(x1, f1), self.efilter2(x2, f2)
                return y1 + y2
            else:
                mup = mup * np.eye(2)
                y1, y2 = self.zconv2S(x1, f1, mup), self.zconv2S(x2, f2, mup)
                return y1 + y2
        if mup.shape == (2, 2):
            y1, y2 = self.zconv2(x1, f1, mup), self.zconv2(x2, f2, mup)
            return y1 + y2

    def zconv2(self, x, h, m):
        h0 = (
            self.rot45(h)
            if sum(sum(m == np.array([[1, -1], [1, 1]]))) == 4
            else self.my_upsamp2df(h, m)
        )
        return torch.conv2d(self.perext(x, h0.size(-2) // 2, h0.size(-1) // 2), h0)

    def my_upsamp2df(self, h0, mup):
        m, n = h0.size()[-2:]
        power = math.log2(mup[0, 0])
        R1, R2 = torch.zeros((1, 1, int(2**power * (m - 1) + 1), m)).to(
            self.device
        ), torch.zeros((1, 1, n, int(2**power * (n - 1) + 1))).to(self.device)
        for i in range(1, m + 1):
            R1[:, :, int((i - 1) * 2 ** (power)), i - 1] = 1
        for i in range(1, n + 1):
            R2[:, :, i - 1, int((i - 1) * 2 ** (power))] = 1
        return torch.matmul(torch.matmul(R1, h0), R2)

    def efilter2(self, x, f):
        return torch.conv2d(self.perext(x, f.size(-2) // 2, f.size(-1) // 2), f)

    def perext(self, x, er, ec):
        x = torch.cat((x[:, :, -er:, :], x, x[:, :, :er, :]), -2)
        return torch.cat((x[:, :, :, -ec:], x, x[:, :, :, :ec]), -1)

    def rot45(self, h0):
        h = torch.zeros(1, 1, 2 * h0.size()[-2] - 1, 2 * h0.size()[-1] - 1).to(
            self.device
        )
        sz1, sz2 = h0.size()[-2:]
        for i in range(1, sz1 + 1):
            r, c = i + np.arange(0, sz2, 1), sz2 - i + np.arange(1, sz2 + 1, 1)
            for j in range(1, sz2 + 1):
                h[:, :, r[j - 1] - 1, c[j - 1] - 1] = h0[:, :, i - 1, j - 1]
        return h

    def nsdfbrec(self, x, dfilter):
        clevels = math.log2(x.size()[1])
        if clevels == 0:
            return x
        k1, k2, f1, f2 = dfilter["0"], dfilter["1"], dfilter["2"], dfilter["3"]
        q1 = np.array([[1, -1], [1, 1]])
        if clevels == 1:
            y = self.nssfbrec(x[:, 0:1, :, :], x[:, 1:2, :, :], k1, k2)
        else:
            for ll in range(int(clevels), 3 - 1, -1):
                for k in range(1, 2 ** (ll - 2) + 1):
                    slk = 2 * math.floor((k - 1) / 2) - 2 ** (ll - 3) + 1
                    mkl = np.matmul(
                        2 * np.array([[2 ** (ll - 3), 0], [0, 1]]),
                        np.array([[1, 0], [-slk, 1]]),
                    )
                    i = (k - 1) % 2 + 1
                    x[:, k - 1 : k, :, :] = self.nssfbrec(
                        x[:, 2 * k - 2 : 2 * k - 1, :, :],
                        x[:, 2 * k - 1 : 2 * k, :, :],
                        f1[str(i - 1)],
                        f2[str(i - 1)],
                        mkl,
                    )
                for k in range(2 ** (ll - 2) + 1, 2 ** (ll - 1) + 1):
                    slk = (
                        2 * math.floor((k - 2 ** (ll - 2) - 1) / 2) - 2 ** (ll - 3) + 1
                    )
                    mkl = np.matmul(
                        2 * np.array([[1, 0], [0, 2 ** (ll - 3)]]),
                        np.array([[1, -slk], [0, 1]]),
                    )
                    i = (k - 1) % 2 + 3
                    x[:, k - 1 : k, :, :] = self.nssfbrec(
                        x[:, 2 * k - 2 : 2 * k - 1, :, :],
                        x[:, 2 * k - 1 : 2 * k, :, :],
                        f1[str(i - 1)],
                        f2[str(i - 1)],
                        mkl,
                    )
            x[:, 0:1, :, :] = self.nssfbrec(
                x[:, 0:1, :, :], x[:, 1:2, :, :], k1, k2, q1
            )
            x[:, 1:2, :, :] = self.nssfbrec(
                x[:, 2:3, :, :], x[:, 3:4, :, :], k1, k2, q1
            )
            y = self.nssfbrec(x[:, 0:1, :, :], x[:, 1:2, :, :], k1, k2)
        return y

    def nsfbrec(self, y0, y1, g0, g1, lev):
        if lev != 0:
            L = 2**lev
            x = torch.conv2d(
                self.symext(
                    y0,
                    (2 ** (lev - 1)) * (g0.size(-2) - 1),
                    (2 ** (lev - 1)) * (g0.size(-1) - 1),
                ),
                g0,
                dilation=L,
            ) + torch.conv2d(
                self.symext(
                    y1,
                    (2 ** (lev - 1)) * (g1.size(-2) - 1),
                    (2 ** (lev - 1)) * (g1.size(-1) - 1),
                ),
                g1,
                dilation=L,
            )
        else:
            x = torch.conv2d(
                self.symext(y0, g0.size(-2) // 2, g0.size(-1) // 2), g0
            ) + torch.conv2d(self.symext(y1, g1.size(-2) // 2, g1.size(-1) // 2), g1)
        return x

    def symext(self, x, er, ec):
        x = torch.cat(
            (torch.flip(x[:, :, :er, :], [-2]), x, torch.flip(x[:, :, -er:, :], [-2])),
            -2,
        )
        return torch.cat(
            (torch.flip(x[:, :, :, :ec], [-1]), x, torch.flip(x[:, :, :, -ec:], [-1])),
            -1,
        )

    def nsctRec(self, y, x):
        y = y[::-1]
        n = len(y)
        xlo = copy.deepcopy(x)
        nIndex = n - 1
        for i in range(1, n + 1):
            if y[i - 1].size()[1] != 1:
                xhi = self.nsdfbrec(y[i - 1], self.filters)
            else:
                xhi = y[i - 1]
            x = self.nsfbrec(xlo, xhi, self.g1, self.g2, nIndex)
            xlo = x
            nIndex = nIndex - 1
        return x

    def atrousfilters(self):
        A = np.array(
            [
                [
                    -1.67551636e-04,
                    -1.00530982e-03,
                    -2.51327454e-03,
                    -3.35103272e-03,
                    -2.51327454e-03,
                    -1.00530982e-03,
                    -1.67551636e-04,
                ],
                [
                    -1.00530982e-03,
                    -5.24666309e-03,
                    -1.19388640e-02,
                    -1.53950215e-02,
                    -1.19388640e-02,
                    -5.24666309e-03,
                    -1.00530982e-03,
                ],
                [
                    -2.51327454e-03,
                    -1.19388640e-02,
                    6.76941007e-02,
                    1.54239380e-01,
                    6.76941007e-02,
                    -1.19388640e-02,
                    -2.51327454e-03,
                ],
                [
                    -3.35103272e-03,
                    -1.53950215e-02,
                    1.54239380e-01,
                    3.32566738e-01,
                    1.54239380e-01,
                    -1.53950215e-02,
                    -3.35103272e-03,
                ],
                [
                    -2.51327454e-03,
                    -1.19388640e-02,
                    6.76941007e-02,
                    1.54239380e-01,
                    6.76941007e-02,
                    -1.19388640e-02,
                    -2.51327454e-03,
                ],
                [
                    -1.00530982e-03,
                    -5.24666309e-03,
                    -1.19388640e-02,
                    -1.53950215e-02,
                    -1.19388640e-02,
                    -5.24666309e-03,
                    -1.00530982e-03,
                ],
                [
                    -1.67551636e-04,
                    -1.00530982e-03,
                    -2.51327454e-03,
                    -3.35103272e-03,
                    -2.51327454e-03,
                    -1.00530982e-03,
                    -1.67551636e-04,
                ],
            ]
        )
        B = np.array(
            [
                [
                    1.67551636e-04,
                    1.00530982e-03,
                    2.51327454e-03,
                    3.35103272e-03,
                    2.51327454e-03,
                    1.00530982e-03,
                    1.67551636e-04,
                ],
                [
                    1.00530982e-03,
                    -1.22542382e-03,
                    -1.39494836e-02,
                    -2.34375000e-02,
                    -1.39494836e-02,
                    -1.22542382e-03,
                    1.00530982e-03,
                ],
                [
                    2.51327454e-03,
                    -1.39494836e-02,
                    -6.76941007e-02,
                    -1.02462685e-01,
                    -6.76941007e-02,
                    -1.39494836e-02,
                    2.51327454e-03,
                ],
                [
                    3.35103272e-03,
                    -2.34375000e-02,
                    -1.02462685e-01,
                    8.48651695e-01,
                    -1.02462685e-01,
                    -2.34375000e-02,
                    3.35103272e-03,
                ],
                [
                    2.51327454e-03,
                    -1.39494836e-02,
                    -6.76941007e-02,
                    -1.02462685e-01,
                    -6.76941007e-02,
                    -1.39494836e-02,
                    2.51327454e-03,
                ],
                [
                    1.00530982e-03,
                    -1.22542382e-03,
                    -1.39494836e-02,
                    -2.34375000e-02,
                    -1.39494836e-02,
                    -1.22542382e-03,
                    1.00530982e-03,
                ],
                [
                    1.67551636e-04,
                    1.00530982e-03,
                    2.51327454e-03,
                    3.35103272e-03,
                    2.51327454e-03,
                    1.00530982e-03,
                    1.67551636e-04,
                ],
            ]
        )
        return torch.from_numpy(A)[None, None, :, :].float().to(
            self.device
        ), torch.from_numpy(B)[None, None, :, :].float().to(self.device)

    def parafilters(self, f1, f2):
        y1, y2 = {}, {}
        y1["0"], y2["0"] = self.modulate2(f1, "r"), self.modulate2(f2, "r")
        y1["1"], y2["1"] = self.modulate2(f1, "c"), self.modulate2(f2, "c")
        y1["2"], y2["2"] = y1["0"].T, y2["0"].T
        y1["3"], y2["3"] = y1["1"].T, y2["1"].T
        for i in range(4):
            y1["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y1["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
            y2["{}".format(i)] = (
                torch.from_numpy(
                    self.resampz(y2["{}".format(i)], i + 1)[None, None, :, :]
                )
                .float()
                .to(self.device)
            )
        return y1, y2

    def resampz(self, x, sampleType):
        sx = x.shape
        if (sampleType == 1) or (sampleType == 2):
            y = np.zeros((sx[0] + sx[1] - 1, sx[1]))
            shift1 = (
                -1 * np.arange(0, sx[1], 1, dtype=int)
                if sampleType == 1
                else np.arange(0, sx[1], 1, dtype=int)
            )
            if shift1[-1] < 0:
                shift1 = shift1 - shift1[-1]
            for n in range(sx[1]):
                y[shift1[n] + np.arange(0, sx[0], 1, dtype=int), n] = x[:, n]
            start, finish = 0, y.shape[0] - 1
            while np.sum(np.abs(y[start, :])) == 0:
                start = start + 1
            while np.sum(np.abs(y[finish, :])) == 0:
                finish = finish - 1
            y = y[start : finish + 1, :]
        else:
            y = np.zeros((sx[0], sx[1] + sx[0] - 1))
            shift2 = (
                -1 * np.arange(0, sx[0], 1, dtype=int)
                if sampleType == 3
                else np.arange(0, sx[0], 1, dtype=int)
            )
            if shift2[-1] < 0:
                shift2 = shift2 - shift2[-1]
            for m in range(sx[0]):
                y[m, shift2[m] + np.arange(0, sx[1], 1, dtype=int)] = x[m, :]
            start, finish = 0, y.shape[1] - 1
            while np.sum(np.abs(y[:, start])) == 0:
                start = start + 1
            while np.sum(np.abs(y[:, finish])) == 0:
                finish = finish - 1
            y = y[:, start : finish + 1]
        return y

    def dfilters(self):
        A = np.array(
            [
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
                [-0.0, -0.125, 0.25, -0.125, -0.0],
                [-0.0625, 0.25, 1.75, 0.25, -0.0625],
                [-0.0, -0.125, 0.25, -0.125, -0.0],
                [-0.0, -0.0, -0.0625, -0.0, -0.0],
            ]
        )
        B = np.array(
            [[-0.0, -0.125, -0.0], [-0.125, 0.5, -0.125], [-0.0, -0.125, -0.0]]
        )
        return A / math.sqrt(2), B / math.sqrt(2)

    def modulate2(self, x, modulateType):
        o = np.floor(np.array(x.shape) / 2) + 1
        n1, n2 = (
            np.arange(1, x.shape[0] + 1, 1) - o[0],
            np.arange(1, x.shape[1] + 1, 1) - o[1],
        )
        if modulateType == "c":
            m2 = (-1) ** n2
            return x * np.repeat(m2[None, :], x.shape[0], axis=0)
        elif modulateType == "r":
            m1 = (-1) ** n1
            return x * np.repeat(m1[:, None], x.shape[1], axis=1)
