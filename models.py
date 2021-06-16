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

    def mapIndex(self, index: int):
        if index == 0:
            return self.gap
        elif index == 1:
            return self.bytes
        elif index == 2:
            return self.sourcePort
        else:
            return self.destinationPort

@dataclass(frozen=True)
class ConnectionKey:
    pcap: str
    sourceIp: str
    destIp: str
    window: int



