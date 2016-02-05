assert(opt, "Options need to be parsed before including the profiler!")

if opt.definition ~= '' then
    -- TODO: what is the convention for definition files? Assuming createModel()
    paths.dofile(path.join('..', opt.definition))
    model = createModel(opt.nGPU)
elseif opt.checkpoint ~= '' then
    require 'nn'
    require 'cunn'
    require 'cudnn'
    require 'nngraph'
    model = torch.load(path.join('..', opt.checkpoint))
else
    assert(false, "Don't know how to load the model")
end

print('Model loaded!')
print('')
