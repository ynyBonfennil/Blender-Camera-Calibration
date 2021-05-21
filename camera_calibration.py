import bpy
import bpy_extras
import numpy
from mathutils import Matrix, Vector
import xml.etree.ElementTree as ET
import json

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------
# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

#---------------------------------------------------------------
# write camera calibration data to file (xml)
#---------------------------------------------------------------
def write_cam_calib_xml(RT, K, file_path):
    root = ET.Element("opencv_storage")
    
    nRT = numpy.matrix(RT)
    nRT = nRT/1000
    str_mat = "%f %f %f %f %f %f %f %f %f %f %f %f"%(nRT[0,0],nRT[0,1],nRT[0,2],nRT[0,3],
                                                        nRT[1,0],nRT[1,1],nRT[1,2],nRT[1,3],
                                                        nRT[2,0],nRT[2,1],nRT[2,2],nRT[2,3])

    CamMat = ET.SubElement(root, "CameraMatrix", {"type_id": "opencv-matrix"})
    CamMat_rows = ET.SubElement(CamMat, "rows")
    CamMat_rows.text = "3"
    CamMat_cols = ET.SubElement(CamMat, "cols")
    CamMat_cols.text = "4"
    CamMat_d = ET.SubElement(CamMat, "dt")
    CamMat_d.text = "d"
    CamMat_data = ET.SubElement(CamMat, "data")
    CamMat_data.text = str_mat
    
    nK = numpy.matrix(K)
    nK = nK/1000
    str_mat = "%f %f %f %f %f %f %f %f %f"%(nK[0,0],nK[0,1],nK[0,2],
                                            nK[1,0],nK[1,1],nK[1,2],
                                            nK[2,0],nK[2,1],nK[2,2])
    
    KMat = ET.SubElement(root, "Intrinsics", {"type_id": "opencv-matrix"})
    KMat_rows = ET.SubElement(KMat, "rows")
    KMat_rows.text = "3"
    KMat_cols = ET.SubElement(KMat, "cols")
    KMat_cols.text = "3"
    KMat_d = ET.SubElement(KMat, "dt")
    KMat_d.text = "d"
    KMat_data = ET.SubElement(KMat, "data")
    KMat_data.text = str_mat
    
    str_mat = "0 0 0 0 0 0 0 0"
    
    DistMat = ET.SubElement(root, "Distortion", {"type_id": "opencv-matrix"})
    DistMat_rows = ET.SubElement(DistMat, "rows")
    DistMat_rows.text = "8"
    DistMat_cols = ET.SubElement(DistMat, "cols")
    DistMat_cols.text = "1"
    DistMat_d = ET.SubElement(DistMat, "dt")
    DistMat_d.text = "d"
    DistMat_data = ET.SubElement(DistMat, "data")
    DistMat_data.text = str_mat
    
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)
    
    # \' cannot be used because of opencv specification
    file_text = ""
    with open(file_path, "r") as f:
        file_text = f.read()
        file_text = file_text.replace("\'", "\"")
    
    with open(file_path, "w") as f:
        f.write(file_text)

#---------------------------------------------------------------
# write camera calibration data to file (json)
#---------------------------------------------------------------
def write_cam_calib_json(RT, K, file_path):
    data = {}
    
    nRT = numpy.matrix(RT)
    nRT = nRT / 1000
    data["RT"] = [nRT[0,0],nRT[0,1],nRT[0,2],nRT[0,3],
                    nRT[1,0],nRT[1,1],nRT[1,2],nRT[1,3],
                    nRT[2,0],nRT[2,1],nRT[2,2],nRT[2,3]]
    
    nK = numpy.matrix(K)
    nK = nK/1000
    data["K"] = [nK[0,0],nK[0,1],nK[0,2],
                nK[1,0],nK[1,1],nK[1,2],
                nK[2,0],nK[2,1],nK[2,2]]
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------------------------------------
if __name__ == "__main__":
    # camer calibration
    for i in range(1,6):
        cam_name = "Camera.00%d"%i
        cam = bpy.data.objects[cam_name]
        P, K, RT = get_3x4_P_matrix_from_blender(cam)

        # save camera calibration data to files 
        nP = numpy.matrix(P)
        write_cam_calib_json(RT, K, "./calibrations/"+cam_name+".json")
    
    # rendering
    for i in range(1,6):
        cam_name = "Camera.00%d"%i
        cam = bpy.data.objects[cam_name]
        scene = bpy.context.scene
        scene.camera = cam
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = "./images/"+cam_name+".png"
        bpy.ops.render.render(write_still = 1)

