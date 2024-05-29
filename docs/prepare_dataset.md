

## PianoMotion10M

Download PianoMotion10M V1.0 full dataset data [HERE](https://drive.google.com/drive/folders/1JY0zOE0s7v9ZYLlIP1kCZUdNrih5nYEt?usp=sharing).

```
cd /path/to/PianoMotion10M_Dataset
unzip annotation.zip
unzip audio.zip
unzip midi.zip
```


**Folder structure**
```
/path/to/PianoMotion10M_Dataset
├── annotation/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1/
│   │   │   ├──BV1f34y1i7U1_seq_0000.json
│   │   │   ├──BV1f34y1i7U1_seq_0001.json
│   │   │   ├──...
│   │   ├── BV1X44y1J7CR/
│   ├── 2084102325/
│   ├── ...
├── audio/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1/
│   │   │   ├──BV1f34y1i7U1_seq_0000.mp3
│   │   │   ├──BV1f34y1i7U1_seq_0001.mp3
│   │   │   ├──...
│   │   ├── BV1X44y1J7CR/
│   ├── 2084102325/
│   ├── ...
├── midi/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1.mid
│   │   ├── BV1X44y1J7CR.mid
│   │   ├── ...
│   ├── 2084102325/
│   ├── ...
├── train.txt
├── test.txt
├── valid.txt
```

**Usage**

`draw.py` shows the usage of our dataset and visualizes some samples of hand motions under `./draw_sample`.

```shell
python draw.py
```