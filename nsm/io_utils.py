from nsm.typing import *

import vtk
import xml.etree.ElementTree as ET

from nsm import field_utils as fu
from vtkmodules.util import numpy_support

VTK_EXT = {
  Grid: "vti",
  SEM: "vts",
}

def dump(path: str, vs: Union[DataType, List[DataType]]):
  """
    Dump a snapshot or a time series into VTK format.
  """
  if isinstance(vs, List):
    os.mkdir(path)

    pvd = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(pvd, "Collection")

    for t, v in enumerate(vs):
      elem = ET.SubElement(collection, "DataSet", timestep=str(t))
      elem.set("file", dump(f"{path}/time.{t}", v))

    with open(file:=f"{path}.pvd", "wb") as fd:
      ET.ElementTree(pvd).write(fd, encoding="utf-8", xml_declaration=True)

    return file

  else:
    file = f"{path}.{VTK_EXT[type(vs)]}"

    if isinstance(vs, Grid):
      data = vtk.vtkImageData()
      data.SetDimensions(*vs.resolution)

      cell_data = jnp.stack(fu.from_grid(vs), axis=-1).ravel()
      cell = numpy_support.numpy_to_vtk(cell_data, deep=1)
      cell.SetNumberOfComponents(vs.ndim)
      cell.SetName("Velocity")

      writer = vtk.vtkXMLImageDataWriter()
      writer.SetInputData(data)
      data.GetCellData().AddArray(cell)

    if isinstance(vs, SEM):
      raise NotImplementedError

    writer.SetFileName(file)
    writer.Write()
    return file
