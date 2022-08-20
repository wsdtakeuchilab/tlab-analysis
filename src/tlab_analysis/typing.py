import io
import os

FilePath = str | os.PathLike[str]
FilePathOrBuffer = FilePath | io.BufferedIOBase
