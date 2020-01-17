import sys
from termcolor import cprint
from subprocess import CalledProcessError
from argparse import ArgumentParser
from itertools import product
from functools import reduce

import numpy as np
import openmc
from pymoab import core, types
from pymoab.scd import ScdInterface
from pymoab.hcoord import HomCoord


def create_model(mesh_dims,
                 estimator='tracklength',
                 external_geom=False,
                 mesh_library='moab'):
    """
    Creates a simple model for testing tallies
    """
    # materials

    materials = openmc.Materials()

    fuel_mat = openmc.Material(name="fuel")
    fuel_mat.add_nuclide("U235", 1.0)
    fuel_mat.set_density('g/cc', 4.5)
    materials.append(fuel_mat)

    zirc_mat = openmc.Material(name="zircaloy")
    zirc_mat.add_nuclide("Zr90", 0.5145)
    zirc_mat.add_nuclide("Zr91", 0.1122)
    zirc_mat.add_nuclide("Zr92", 0.1715)
    zirc_mat.add_nuclide("Zr94", 0.1738)
    zirc_mat.add_nuclide("Zr96", 0.028)
    zirc_mat.set_density("g/cc", 5.77)
    materials.append(zirc_mat)

    water_mat = openmc.Material(name="water")
    water_mat.add_nuclide("H1", 2.0)
    water_mat.add_nuclide("O16", 1.0)
    water_mat.set_density("atom/b-cm", 0.07416)
    materials.append(water_mat)

    materials.export_to_xml()

    # bounding surfaces

    fuel_min_x = openmc.XPlane(x0=-5.0, name="minimum x")
    fuel_max_x = openmc.XPlane(x0=5.0, name="maximum x")

    fuel_min_y = openmc.YPlane(y0=-5.0, name="minimum y")
    fuel_max_y = openmc.YPlane(y0=5.0, name="maximum y")

    fuel_min_z = openmc.ZPlane(z0=-5.0, name="minimum z")
    fuel_max_z = openmc.ZPlane(z0=5.0, name="maximum z")

    fuel_cell = openmc.Cell(name="fuel")
    fuel_cell.region = +fuel_min_x & -fuel_max_x & \
                       +fuel_min_y & -fuel_max_y & \
                       +fuel_min_z & -fuel_max_z
    fuel_cell.fill = fuel_mat

    clad_min_x = openmc.XPlane(x0=-6.0, name="minimum x")
    clad_max_x = openmc.XPlane(x0=6.0, name="maximum x")

    clad_min_y = openmc.YPlane(y0=-6.0, name="minimum y")
    clad_max_y = openmc.YPlane(y0=6.0, name="maximum y")

    clad_min_z = openmc.ZPlane(z0=-6.0, name="minimum z")
    clad_max_z = openmc.ZPlane(z0=6.0, name="maximum z")

    clad_cell = openmc.Cell(name="clad")
    clad_cell.region = (-fuel_min_x | +fuel_max_x | \
                        -fuel_min_y | +fuel_max_y | \
                        -fuel_min_z | +fuel_max_z) & \
                        (+clad_min_x & -clad_max_x & \
                         +clad_min_y & -clad_max_y & \
                         +clad_min_z & -clad_max_z)
    clad_cell.fill = zirc_mat

    if external_geom:
        bounds = (15, 15, 15)
    else:
        bounds = (10, 10, 10)

    water_min_x = openmc.XPlane(x0=-bounds[0], name="minimum x", boundary_type='vacuum')
    water_max_x = openmc.XPlane(x0=bounds[0], name="maximum x", boundary_type='vacuum')

    water_min_y = openmc.YPlane(y0=-bounds[1], name="minimum y", boundary_type='vacuum')
    water_max_y = openmc.YPlane(y0=bounds[1], name="maximum y", boundary_type='vacuum')

    water_min_z = openmc.ZPlane(z0=-bounds[2], name="minimum z", boundary_type='vacuum')
    water_max_z = openmc.ZPlane(z0=bounds[2], name="maximum z", boundary_type='vacuum')

    water_cell = openmc.Cell(name="water")
    water_cell.region = (-clad_min_x | +clad_max_x | \
                         -clad_min_y | +clad_max_y | \
                         -clad_min_z | +clad_max_z) & \
                         (+water_min_x & -water_max_x & \
                          +water_min_y & -water_max_y & \
                          +water_min_z & -water_max_z)
    water_cell.fill = water_mat

    # create a containing universe

    root_univ = openmc.Universe()
    root_univ.add_cells([fuel_cell, clad_cell, water_cell])

    geom = openmc.Geometry(root=root_univ)

    geom.export_to_xml()

    ### Tallies ###

    # create meshes

    coarse_mesh = openmc.RegularMesh()
    coarse_mesh.dimension = mesh_dims
    coarse_mesh.lower_left = (-10.0, -10.0, -10.0)
    coarse_mesh.upper_right = (10.0, 10.0, 10.0)

    coarse_filter = openmc.MeshFilter(mesh=coarse_mesh)

    fine_mesh = openmc.RegularMesh()
    fine_mesh.dimension = (40, 40, 40)
    fine_mesh.lower_left = (-10.0, -10.0, -10.0)
    fine_mesh.upper_right = (10.0, 10.0, 10.0)

    fine_filter = openmc.MeshFilter(mesh=fine_mesh)

    uscd_mesh = openmc.UnstructuredMesh()
    if mesh_library == 'moab':
        uscd_mesh.filename = 'test_mesh_tets.h5m'
    else:
        uscd_mesh.filename = 'test_mesh_tets.exo'
        uscd_mesh.mesh_lib = 'libmesh'
    mb = core.Core()
    mb.load_file(uscd_mesh.filename)
    all_tets = mb.get_entities_by_type(0, types.MBTET)
    uscd_mesh.size = all_tets.size()

    uscd_filter = openmc.MeshFilter(mesh=uscd_mesh)

    # create tallies
    tallies = openmc.Tallies()

    coarse_mesh_tally = openmc.Tally(name="coarse mesh tally")
    coarse_mesh_tally.filters = [coarse_filter]
    coarse_mesh_tally.scores = ['flux']
    coarse_mesh_tally.estimator = estimator
    tallies.append(coarse_mesh_tally)

    fine_mesh_tally = openmc.Tally(name="fine mesh tally")
    fine_mesh_tally.filters = [fine_filter]
    fine_mesh_tally.scores = ['flux']
    fine_mesh_tally.estimator = estimator
    tallies.append(fine_mesh_tally)

    uscd_tally = openmc.Tally(name="unstructured mesh tally")
    uscd_tally.filters = [uscd_filter]
    uscd_tally.scores = ['flux']
    uscd_tally.estimator = estimator
    tallies.append(uscd_tally)

    tallies.export_to_xml()

def hex_to_tets(mb, h, make_tris=False):

    # lookup the hex vertices
    hex_verts = mb.get_connectivity(h)

    # get all 2-D elements adjacent to this hex

    faces = mb.get_adjacencies(h, 2, types.UNION)

    quad_tri_handle = mb.tag_get_handle("QUAD_TRI")

    tris = mb.tag_get_data(quad_tri_handle, faces, flat=True)

    # get the center of the hex element
    verts = mb.get_adjacencies(h, 0)
    coords = mb.get_coords(verts)
    coords.shape = (len(verts), 3)
    center = np.mean(coords, axis=0)

    # create a vertex there
    center_vert = mb.create_vertices(center)[0]

    # create a tet for each triangle
    for tri in tris:
        tri_conn = mb.get_connectivity(tri)

        mb.create_element(types.MBTET, (tri_conn[0], tri_conn[1], tri_conn[2], center_vert))

def quad_to_tris(mb, q):
    conn = mb.get_connectivity(q)
    tri1 = mb.create_element(types.MBTRI, (conn[0], conn[1], conn[2]))
    tri2 = mb.create_element(types.MBTRI, (conn[2], conn[3], conn[0]))

    quad_tri_handle = mb.tag_get_handle("QUAD_TRI",
                                        2,
                                        types.MB_TYPE_HANDLE,
                                        types.MB_TAG_SPARSE,
                                        create_if_missing=True)

    mb.tag_set_data(quad_tri_handle, q, [tri1, tri2])

def create_plots():

    plot = openmc.Plot()
    plot.origin = (0.0, 0.0, 0.0)
    plot.width = (20, 20)
    plot.height = (200, 200)
    plot.color_by = 'material'

    plots = openmc.Plots()
    plots.append(plot)

    plots.export_to_xml()

def create_settings(batches, particles):
    """
    Create settings.xml file
    """

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = particles
    settings.batches = batches

    # source setup
    r = openmc.stats.Uniform(a=0.0, b=0.0)
    theta = openmc.stats.Discrete(x=[0.0], p=[1.0])
    phi = openmc.stats.Discrete(x=[0.0], p=[1.0])
    origin = (1.0, 1.0, 1.0)

    space = openmc.stats.SphericalIndependent(r=r,
                                              theta=theta,
                                              phi=phi,
                                              origin=origin)

    angle = openmc.stats.Monodirectional((-1.0, 0.0, 0.0))

    energy = openmc.stats.Discrete(x=[15.0E6], p=[1.0])

    source = openmc.Source(space=space, energy=energy, angle=angle)

    settings.source = source

    settings.export_to_xml()

def create_unstructured_mesh(mesh_dims, holes=()):

    mb = core.Core()

    llc = np.array((-10, -10, -10))
    urc = np.array((10, 10, 10))
    dims = np.array(mesh_dims)

    scd = ScdInterface(mb)

    coords = np.ndarray(dims)

    dx, dy, dz = (urc - llc) / dims

    xs, ys, zs = tuple(np.linspace(llc[i], urc[i], dims[i] + 1) for i in range(3))

    coords = []

    for k in zs:
        for j in ys:
            for i in xs:
                coords += [i,j,k]

    llc = HomCoord((0, 0, 0))
    urc = HomCoord(tuple(dims))

    scd.construct_box(llc,
                      urc,
                      coords)

    mb.write_file("test_mesh_hexes.h5m")

    # get all the hexes in the mesh
    all_hexes = mb.get_entities_by_type(0, types.MBHEX)
    holy_hexes = [all_hexes[hole] for hole in holes]
    mb.delete_entities(holy_hexes)

    all_hexes = mb.get_entities_by_type(0, types.MBHEX)
    all_quads = mb.get_adjacencies(all_hexes, 2, create_if_missing=True, op_type=types.UNION)

    for quad in all_quads:
        quad_to_tris(mb, quad)

    for h in all_hexes:
        hex_to_tets(mb, h)

    all_tris = mb.get_entities_by_type(0, types.MBTRI)

    # cleanup un-needed elements
    mb.delete_entities(all_hexes)
    mb.delete_entities(all_tris)
    mb.delete_entities(all_quads)

    all_tets = mb.get_entities_by_type(0, types.MBTET)
    all_tris = mb.get_adjacencies(all_tets,
                                  2,
                                  create_if_missing=True,
                                  op_type=types.UNION)

    mb.delete_entities(all_tris)

    mb.write_file("test_mesh_tets.h5m")
    mb.write_file("test_mesh_tets.exo")

def compare_results(statepoint, holes=()):

    sp = openmc.StatePoint(statepoint)

    for tally in sp.tallies.values():

        if 'fine' in tally.name:
            continue

        if tally.contains_filter(openmc.MeshFilter):

            flt = tally.find_filter(openmc.MeshFilter)

            if isinstance(flt.mesh, openmc.UnstructuredMesh):
                ucd_data, ucd_err = report_unstructured_mesh(tally)
            else:
                reg_data, reg_err = report_structured_mesh(tally, holes)

    # display particles/sec
    particles = sp.n_particles * sp.n_batches # n particles
    runtime = sp.runtime['simulation']
    particles_per_second = particles / runtime
    print("Particles/sec: {}".format(particles_per_second))

    # display FOM for the unstructured mesh
    rel_err = np.nan_to_num(ucd_err / ucd_data, nan=0.0)
    rel_err_sum = np.sum(rel_err)
    fom_time = 1.0 / (runtime * rel_err_sum ** 2)
    fom_particle = 1.0 / (particles * rel_err_sum ** 2)

    print("FOM (time): {}".format(fom_time))
    print("FOM (particles): {}".format(fom_particle))

    # successively check how many decimals the results are equal to
    decimals = 1
    while True:
        try:
            np.testing.assert_array_almost_equal(ucd_data, reg_data, decimals)
        except AssertionError as ae:
            print(ae)
            print()
            break
        # increment decimals
        decimals += 1

    print("Results equal to within {} decimal places.\n".format(decimals))
    if decimals < 5:
        cprint("FAIL - results not equal to within 5 decimals", 'red')
    else:
        cprint("PASS", 'green')

def report_structured_mesh(tally, holes=(), verbose=0):
    # get the tally results
    data = tally.get_reshaped_data(value='mean')
    err = tally.get_reshaped_data(value='std_dev')

    if holes:
        data = np.delete(data, holes)
        err = np.delete(err, holes)

    if verbose:
        print()
        print(tally)
        print("Num bins: {}".format(data.size))
        print("Total score: {}".format(np.sum(data)))
        # print all bins
        if verbose > 1:
            for v, e in zip(data, err):
                print("Val: {}, Std. Dev.: {}".format(np.sum(v),
                                                      np.sum(e)))
        print()

    data = data.reshape(data.size, 1)
    err = err.reshape(err.size, 1)
    return np.sum(data, axis=1), np.sum(err, axis=1)

def report_unstructured_mesh(tally, verbose=0):

    tets_per_hex = 12

    # get the tally results
    data = tally.get_reshaped_data(value='mean')
    data.shape = (data.size // tets_per_hex, tets_per_hex)
    err = tally.get_reshaped_data(value='std_dev')
    err.shape = (err.size // tets_per_hex, tets_per_hex)

    if verbose:
        print()
        print(tally)
        print("Num bins: {}".format(data.shape[0]))
        print("Total score: {}".format(np.sum(data)))
        # print all bins
        if verbose > 1:
            for v, e in zip(data, err):
                print("Val: {}, Std. Dev.: {}".format(np.sum(v),
                                                      np.sqrt(np.sum(e**2))))
        print()

    return np.sum(data, axis=1), np.sum(err, axis=1)

def perform_comparison(mesh_dims=None,
                       with_holes=False,
                       estimator='tracklength',
                       mesh_lib='moab',
                       skip_run=False,
                       external_geom=False,
                       verbose=0,
                       particles=50000,
                       n_threads=15):

    if mesh_dims is None:
        mesh_dims = (np.random.randint(2, 10),
                     np.random.randint(2, 10),
                     np.random.randint(2, 10))

    print("Test Summary:")
    print("Mesh Dimensions: {}".format(mesh_dims))
    print("Estimator: {}".format(estimator))
    print("External Geometry: {}".format(external_geom))
    print("With Holes: {}".format(with_holes))
    print("Mesh Library: {}".format(mesh_lib))
    print("Particles per Batch: {}".format(particles))

    if with_holes:
        # randomly select some holes in the mesh
        holes_ijk = []
        for i in range(reduce(lambda x,y: x*y, mesh_dims) // 10):
            holes_ijk.append((np.random.randint(0, mesh_dims[0] - 1),
                              np.random.randint(0, mesh_dims[1] - 1),
                              np.random.randint(0, mesh_dims[2] - 1)))
    else:
        holes_ijk = ()

    nx, ny, nz = mesh_dims
    holes = tuple({i + j * nx + k * (nx * ny) for i, j, k in holes_ijk})
    create_unstructured_mesh(mesh_dims, holes)

    create_model(mesh_dims, estimator, external_geom, mesh_library=mesh_lib)
    batches = 50
    create_settings(batches, particles)
    create_plots()

    # run openmc if needed
    if not skip_run:
        try:
            openmc.run(threads=n_threads, output=verbose)
        except CalledProcessError:
            cprint("Error running OpenMC. Rerunning with output.", 'red')
            openmc.run(threads=n_threads)

    statepoint_filename = "statepoint.{}.h5".format(batches)

    compare_results(statepoint_filename, holes)

if __name__ == "__main__":

    estimators = ('collision', 'tracklength')
    mesh_libs = ('moab', 'libmesh')

    ap = ArgumentParser(description="A Python program for testing OpenMC" \
                        " unstructured mesh capability.")

    ap.add_argument('-n', dest='skip_run', type=bool, default=False,
                    help="Do not run OpenMC if present")
    ap.add_argument('-v', dest='verbose', type=bool, default=False,
                    help="Enable verbose output from OpenMC runs if present")
    ap.add_argument('-e', dest='estimators', nargs='+', default=estimators,
                    help="Specify estimators to test (collision/tracklength)")
    ap.add_argument('-l', dest='libraries', nargs='+', default=mesh_libs,
                    help="Specify mesh libraries to test")
    ap.add_argument('-p', dest='particles', type=int, default=50000,
                    help="Number of particles per batch")
    ap.add_argument('-t', dest='threads', nargs='+', default=15,
                    help="Number of threads to run OpenMC with")

    ext_geom = (False, True)
    w_holes = (False, True)

    args = ap.parse_args()

    # perform comparisions using the mesh
    for lib in args.libraries:

        print("======================")
        print("{} tests".format(lib))
        print("======================")

        for estimator in args.estimators:

            print("-----------------")
            print("{} estimator".format(estimator))
            print("-----------------")

            for ext, holes in product(ext_geom, w_holes):

                perform_comparison(estimator=estimator,
                                   with_holes=holes,
                                   skip_run=args.skip_run,
                                   external_geom=ext,
                                   mesh_lib=lib,
                                   verbose=args.verbose,
                                   particles=args.particles,
                                   n_threads=args.threads)
