# Subset of model/util.py from V. S. Fonov's py_deep_seg, used with permission.
# Only the symbols needed for inference and for unpickling the ensemble
# weights are kept; function bodies are unchanged from the original.
import math

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class PreprocessModule(nn.Module):
    """
    Simple preprocessing module
    will subtract mean and divide by sd
    """
    def __init__(self, sample_mean, sample_sd):
        """
        Arguments:
            sample_mean - torch tensor broadcastable to the number of input channels
            sample_sd - torch tensor broadcastable to the number of input channels
        """
        super(PreprocessModule, self).__init__()
        sample_mean=torch.nn.Parameter(sample_mean, requires_grad=False)
        sample_sd=torch.nn.Parameter(sample_sd, requires_grad=False)
        self.register_parameter("sample_mean",sample_mean)
        self.register_parameter("sample_sd",sample_sd)

    def forward(self, x):
        x=x.sub(self.sample_mean)
        x=x.div(self.sample_sd)
        return x


def import_parameters(model,to_load,map_location=None):
    """
    load parameters from previously trained model,
    skip initializing missing or non-matching ones
    main purpose: to use pre-trained models
    """
    loaded = torch.load(to_load, map_location=map_location)
    #assert() TODO: make sure it's a state dict
    if isinstance(model,DDP):
        _model=model.module
    else:
        _model=model

    # debug
    cnt=0
    for k in _model.state_dict().keys():
        if k in loaded.keys() and \
            _model.state_dict()[k].size() == loaded[k].size():
            _model.state_dict()[k].copy_(loaded[k])
            cnt+=1
        else:
            #### DEBUG
            print("model import:ignored key:",k)
    print(f"Imported {cnt} of {len(_model.state_dict().keys())} parameters")
    return model


def load_model(model, to_load, map_location=None):
    """
    load a previously trained model.
    """
    to_load = torch.load(to_load, map_location=map_location)

    if isinstance(to_load, nn.Module):
        #old style , saving the whole model
        # check parameters sizes
        model_params   = set(model.state_dict().keys())
        to_load_params = set(to_load.state_dict().keys())

        assert model_params == to_load_params, (model_params - to_load_params, to_load_params - model_params)
        ### TEMPORARY HACK

        # copy saved parameters
        for k in model.state_dict().keys():
        # HACK
        #if k in to_load.state_dict():
            if model.state_dict()[k].size() != to_load.state_dict()[k].size():
                raise Exception("Expected tensor {} of size {}, but got {}".format(
                    k, model.state_dict()[k].size(),
                    to_load.state_dict()[k].size()
                ))
            model.state_dict()[k].copy_(to_load.state_dict()[k])
    else:
        # it's a state_dict
        print("Loading model state dict...")
        if isinstance(model,DDP):
            model.module.load_state_dict(to_load)
        else:
            model.load_state_dict(to_load)


def segment_with_patches_overlap(
        dataset, model,
        use_cuda=True,
        crop=0,
        patch_sz = None,
        stride = None,
        bck = 0,
        out_fuzzy=False,
        out_vae  =False,
        out_latent_vectors=False,
        loc=None,prec=None):
    """
    Apply model to dataset of arbitrary size
    Arguments:
        dataset - torch.Tensor of input data, 5D
        model - torch model
    Keyword arguments:
        use_cuda - use CUDA for inference
        crop - crop patches by this many voxels in all spatial dimensions for output
        patch_sz - size of patch to process with a model
        stride - step between patches, patches can overlap
        bck - background value, to be used for areas where model was not applied
        out_fuzzy - output fuzzy results instead of just discrete
        out_latent_vectors - output internal latent vectors
        loc  - latent vectors location
        prec - latent vectors precision matrix (inv covariance)
    Returns:
        3D segmentation , if out_fuzzy is False
        tuple: 3D segmentation, 4D fuzzy output if out_fuzzy is True
        tuple: 3D segmentation, 3D -log likelyhood if loc and prec are given
    """
    dsize = dataset.size()
    output_size = list( dsize )
    output_size_fuzzy = list( dsize )
    output_size_vae   = list( dsize )
    output_size_likelihood = list( dsize )

    output_size_likelihood[1]=1
    output_size[1] = 1

    output_fuzzy  = None
    output_vae    = None
    output_weight = torch.zeros( output_size )
    output_likelihood = None

    if patch_sz is None: # backward compatibility
        patch_sz = model.patch_sz

    patch_sz_ = patch_sz - crop*2

    if stride is None:
        stride = patch_sz_

    out_roi = [ dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2 ]
    ones = torch.ones([1,1,patch_sz_,patch_sz_,patch_sz_])
    latent_vectors = []

    if loc is not None and prec is not None:
        # convert to expected format
        _loc  = loc.reshape(1, loc.shape[0]).to(torch.double)
        _prec = prec.reshape(1,prec.shape[0],prec.shape[1]).to(torch.double)
        # https://stats.stackexchange.com/questions/97408/relation-of-mahalanobis-distance-to-log-likelihood
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        # import math
        # here we replaced the determinant of covariance matrix with
        # inverse of the determinat of the precision matrix (since it't inverse of covariance matrix)
        _c = -0.5*math.log(float(torch.linalg.det(prec))) + prec.shape[0]*math.log(2*math.pi)
        print('_c',_c)


    # TODO:
    with torch.no_grad():
        for k in range(math.ceil( (out_roi[0]-crop)/stride )):
            for l in range(math.ceil( (out_roi[1]-crop)/stride )):
                for m in range(math.ceil( (out_roi[2]-crop)/stride )):

                    c = [k*stride + crop, l*stride + crop, m*stride + crop]

                    for i in range(3):
                        c[i] = max(min(c[i], dsize[i+2] - patch_sz + crop - 1),crop)
                    # extract a patch
                    in_data = dataset[:, :, c[0]-crop: c[0]-crop+patch_sz, c[1]-crop: c[1]-crop+patch_sz, c[2]-crop: c[2]-crop+patch_sz]

                    if use_cuda:
                        in_data = in_data.cuda()

                    out_ = model.forward(in_data)
                    out = out_['seg']
                    if 'vae' in out_:
                        vae =  out_['vae']
                    else:
                        vae = None

                    if out_latent_vectors and 'latent' in out_:
                        latent_vectors += [out_['latent'].cpu()]

                    patch_output = torch.log_softmax(out,1)

                    if use_cuda:
                        patch_output = patch_output.cpu()

                    if output_fuzzy is None:
                        # need to allocate depending on the number of output classses
                        output_size_fuzzy[1] = patch_output.size()[1]
                        output_fuzzy = torch.zeros( *output_size_fuzzy )

                    if out_vae and vae is not None:
                        if output_vae is None:
                            # need to allocate depending on the number of output classses
                            output_size_vae[1] = vae.size()[1]
                            output_vae = torch.zeros( *output_size_vae )

                        if use_cuda:
                            vae=vae.cpu()

                        output_vae[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += \
                            vae[:, :, crop: crop+patch_sz_, crop: crop+patch_sz_, crop: crop+patch_sz_]

                    if loc is not None and prec is not None:
                        if output_likelihood is None:
                            output_likelihood= torch.zeros( *output_size_likelihood )

                        if 'latent' in out_:

                            lat=out_['latent']
                            # calculate log-likelyhood of latent vector being from N-d gaussian distribution of latent vectors
                            #print("lat:",lat.shape)
                            _lat=lat.reshape(lat.shape[0],lat.shape[1],-1).transpose(1,2).reshape(-1,lat.shape[1]).to(torch.double)

                            #print("_lat:",_lat.shape,'loc:',_loc.shape)
                            _diff=(_lat-_loc).reshape(-1,1,lat.shape[1])
                            #print("_diff:",_diff.shape)
                            # now we have dist calculated by n_batch*n_vox:
                            mahalanobis_dist2=torch.matmul(torch.matmul(_diff,_prec),_diff.transpose(1,2))
                            #print("mahalanobis_dist2:",mahalanobis_dist2.shape)
                            log_likelihood = 0.5*mahalanobis_dist2 + _c
                            # TODO: fix this?
                            #log_likelihood = torch.sqrt(mahalanobis_dist2)
                            # reshape back into a patch
                            log_likelihood = log_likelihood.reshape(lat.shape[0],1,lat.shape[2],lat.shape[3],lat.shape[4]).to(torch.float)
                            # choose resample function
                            log_likelihood = nn.functional.interpolate(
                                                        log_likelihood,
                                                        size=patch_output.shape[2:5],  mode='trilinear', align_corners=False) # TODO
                            if use_cuda:
                                log_likelihood=log_likelihood.cpu()

                            output_likelihood[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += \
                                log_likelihood[:, :, crop: crop+patch_sz_, crop: crop+patch_sz_, crop: crop+patch_sz_]

                    # accumulate data
                    output_fuzzy[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += \
                         patch_output[:, :, crop: crop+patch_sz_, crop: crop+patch_sz_, crop: crop+patch_sz_]

                    output_weight[:, :, c[0]: c[0]+patch_sz_, c[1]: c[1]+patch_sz_, c[2]: c[2]+patch_sz_] += \
                         ones

        # aggregate weights
        #output_weight_save = output_weight.clone().detach()

        invalid = output_weight<1.0

        output_weight.masked_fill_(invalid, 1.0 )

        output_fuzzy /= output_weight


        output_fuzzy = nn.functional.softmax(output_fuzzy, 1)

        # set BG to 1 where mask was invalid
        output_fuzzy[:,0:1,:,:,:].masked_fill_(invalid, 1.0 )
        output_fuzzy[:,1:,:,:,:].masked_fill_(invalid, 0.0 )

        output_dataset = output_fuzzy.max(1)[1].cpu().unsqueeze_(1)
        output_dataset.masked_fill_(invalid, bck )

        if out_vae and output_vae is not None:
            output_vae /= output_weight

        if output_likelihood is not None:
            output_likelihood /= output_weight
        if out_fuzzy or out_vae or out_latent_vectors:
            return output_dataset, output_fuzzy, output_vae, latent_vectors
        elif loc is not None and prec is not None:
            return output_dataset, torch.exp(-1.0*output_likelihood)
        else:
            return output_dataset #, output_weight_save
