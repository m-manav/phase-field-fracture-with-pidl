from config import *

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE/Path('source')))

from field_computation import FieldComputation
from construct_model import construct_model
from input_data_from_mesh import prep_input_data
from plotting import plot_mesh, plot_field, img_plot, plot_energy



# prescribe the index of disp to generate plot
disp_idx = 20

# run as: python .\generate_figs.py hidden_layers neurons seed activation init_coeff
# for example: python .\generate_figs.py 8 400 1 TrainableReLU 3.0


device = 'cpu'
## ############################################################################
## Model construction #########################################################
## ############################################################################
pffmodel, matprop, network = construct_model(PFF_model_dict, mat_prop_dict, 
                                             network_dict, domain_extrema, device)
field_comp = FieldComputation(net = network,
                              domain_extrema = domain_extrema, 
                              lmbda = torch.tensor([0.0], device = device), 
                              theta = loading_angle, 
                              alpha_constraint = numr_dict["alpha_constraint"])
field_comp.net = field_comp.net.to(device)
field_comp.domain_extrema = field_comp.domain_extrema.to(device)
field_comp.theta = field_comp.theta.to(device)


# Prepare input data
inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, 
                                                         mesh_file=fine_mesh_file, device=device)


## ############################################################################
## Setting up fig directory ###################################################
## ############################################################################
if Path.is_dir(model_path):
    figfiles = model_path/Path('figfiles')
    figfiles.mkdir(parents=True, exist_ok=True)
    pngfigs = figfiles/Path('pngfigs')
    pngfigs.mkdir(parents=True, exist_ok=True)
    pdffigs = figfiles/Path('pdffigs')
    pdffigs.mkdir(parents=True, exist_ok=True)
    figdir = {"png": pngfigs, "pdf": pdffigs}

    print(f"tensorboard logdir = {model_path/Path('TBruns')}")

    # plot mesh
    plot_mesh(mesh_file=fine_mesh_file, figdir=figdir)

    # plot initial phase field
    plot_field(inp, hist_alpha, T_conn, figname='Initial-phase-field', figdir=figdir)


    # generate fields at prescribed displacement = disp[disp_idx]
    model = trainedModel_path/Path('trained_1NN_'+str(disp_idx)+'.pt')
    if Path.is_file(model):
        print(f"generating plots for prescribed displacement: {disp[disp_idx]}")
        field_comp.net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        field_comp.lmbda = torch.tensor(disp[disp_idx]).to(device)
        img_plot(field_comp, pffmodel, matprop, inp, T_conn, area_T, figdir, dpi=600)
    else:
        print(f"No trained network available with filename: {model}")


    # plot energy vs prescribed displacement
    plot_energy(field_comp, disp, pffmodel, matprop, inp, T_conn, area_T, trainedModel_path, figdir)
