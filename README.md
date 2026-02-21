# Trustpilot Backend challenge
Problem definition [here](https://followthewhiterabbit.trustpilot.com/cs/step3.html)

# Prerequisites

### Linux
```bash
# Debian/Ubuntu
sudo apt install build-essential cmake

# Fedora
sudo dnf install gcc make cmake
```

For GPU support, also install the OpenCL development package:
```bash
sudo apt install ocl-icd-opencl-dev   # Debian/Ubuntu
sudo dnf install ocl-icd-devel        # Fedora
```
This provides the OpenCL headers and ICD loader library. Your GPU driver (NVIDIA/AMD) provides the runtime, but CMake needs these dev files to enable GPU support at compile time.

### macOS
```bash
xcode-select --install
brew install cmake
```
Metal backend is used automatically on Apple Silicon. OpenCL is available but deprecated.

# Latest Benchmarks

### Mac M2 Max
```
Probing cruncher backends:
  metal[0]: Apple M2 Max
  metal: 1 instance(s)
  opencl: skipped (native GPU backend already active)
  cpu: skipped (accelerated backend already active)
1 cruncher instance(s) total

searching through anas up to 8 words                                
dff9683b844616db3e6d9d27f59f99e9:  lisp wu not statutory
665e5bcb0c20062fe8abaaf4628bb154:  wu lisp not statutory
7dbf96a1e3ee1a109eb0ad7e9fec8adc:  printouts ty outlaws
23170acc097c24edb98fc5488ab033fe:  ty outlaws printouts
f1372a01b8bdb225c9217724439f833f:  sort tutu at wi pylons
dea83b3a70028599b1c81f342caff836:  wry tints output also
d89766a5a822578b98c0d2b1745399ff:  til yawns tot up routs
214da12103aa091eb6681b5e813f3714:  rusty walnut it so top
68146baaed010d17715db43f4368f87b:  stout yawls printout
e4820b45d2277f3844eac66c903e84be:  printout stout yawls
0f020290925166ba0de4b20a84732810:  tutor yup tin was lost
2353c0e120276849118a6c88a8193dba:  tas ur nut lo wi top sty
e881e4faccf6322a30f3517fab5de695:  i put narowly st to stu
4269de5d2cf8014eb7a61ebd75586312:  ail no or pt st stu ty wu
2a856509c4a38603de1c8e6ab8ebfa06:  wo a st to triply nut us
2d287638fba2f941a61ac6f699db5121:  ail no ort pt st ty us wu
a3d7911494e1fdabd394e6980efaa63a:  ail no or put st st ty wu
c6488ef7ae6b03e4c6d58623280b5c7b:  ail no ort pu st st ty wu
4e297caf3ea38849c54de18f67d1d1a1:  ail no opt ru st st ty wu

  metal(1): 9.9T anas, 2.4GAna/s avg
  total: 9.9T anas in 4593.0s, 2.4GAna/s effective
```
