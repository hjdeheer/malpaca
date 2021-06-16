import pickle
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageInfo:
    __slots__ = ['gap', 'bytes', 'sourcePort', 'destinationPort', 'connectionLabel']
    gap: int
    bytes: int
    sourcePort: int
    destinationPort: int
    connectionLabel: str

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


def readPcapPkl(file) -> dict[tuple[str, str], list[PackageInfo]]:
    with open(file, 'rb') as file:
        connections = pickle.load(file)

    return connections


def testRead():
    connections = readPcapPkl('pcaps/CTU-Honeypot-Capture-4-1.pcap.pkl')

    for k, v in connections.items():
        print(k)
        print(len(v))



testRead()