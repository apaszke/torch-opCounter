require 'cutorch'

cmd = torch.CmdLine()
cmd:text()
cmd:text('A network profiler')
cmd:text()
cmd:text('Options')
cmd:option('-checkpoint','','path to model checkpoint')
cmd:option('-definition','','path to model definition')
cmd:option('-inputSize','1x3x224x224','input tensor dimensions')
cmd:option('-MACs',false,'use multiply-adds when counting ops')
cmd:option('-nGPU',cutorch.getDeviceCount(),'input tensor dimensions')
cmd:text()

opt = cmd:parse(arg or {})

if opt.checkpoint == '' and opt.definition == '' then
    print('No model specified!')
    os.exit(false)
end

if opt.checkpoint ~= '' and opt.definition ~= '' then
    print('Specify only the checkpoint or only the definition file!')
    os.exit(false)
end

-- parse inputSize
local inputSize = {}
for i, s in pairs(opt.inputSize:split('x')) do
    inputSize[i] = tonumber(s)
end
opt.inputSize = inputSize

