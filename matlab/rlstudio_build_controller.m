function rlstudio_build_controller(outSlx, specJson)
%RLSTUDIO_BUILD_CONTROLLER Construit un modele Simulink contenant un controleur extrait par RL Studio.
%   outSlx   : chemin du modele a creer (ecrase s'il existe)
%   specJson : fichier JSON {obs_names, act_names, code, sample_time, description}
%              code = texte d'une fonction MATLAB `u = fcn(obs)` (compatible generation de code) ;
%              utilise aussi pour les modeles reduits identifies : entrees = obs_names, sorties = act_names
%   Structure : Inport (un par observation) -> Mux -> bloc MATLAB Function -> Demux -> Outport (un par action).
spec = jsondecode(fileread(specJson));
obsNames = cellstr(spec.obs_names);
actNames = cellstr(spec.act_names);
n = numel(obsNames);
m = numel(actNames);
[~, mdl] = fileparts(outSlx);
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
load_system(mdl);

% entrees -> Mux
mux = [mdl '/ObsMux'];
add_block('simulink/Signal Routing/Mux', mux, 'Inputs', num2str(n), ...
    'Position', [180 40 185 40 + 50 * max(n, 2)]);
for k = 1:n
    blk = [mdl '/' obsNames{k}];
    add_block('simulink/Sources/In1', blk, 'Position', [40 40 + 50 * (k - 1) 70 54 + 50 * (k - 1)]);
    add_line(mdl, [obsNames{k} '/1'], sprintf('ObsMux/%d', k));
end

% controleur : bloc MATLAB Function
fcn = [mdl '/Controller'];
add_block('simulink/User-Defined Functions/MATLAB Function', fcn, 'Position', [260 40 380 40 + 50 * max(n, 2)]);
chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', fcn);
chart.Script = spec.code;
if spec.sample_time > 0
    chart.SampleTime = num2str(spec.sample_time, 17);
end
add_line(mdl, 'ObsMux/1', 'Controller/1');

% sorties
if m == 1
    add_block('simulink/Sinks/Out1', [mdl '/' actNames{1}], 'Position', [460 60 490 74]);
    add_line(mdl, 'Controller/1', [actNames{1} '/1']);
else
    add_block('simulink/Signal Routing/Demux', [mdl '/ActDemux'], 'Outputs', num2str(m), ...
        'Position', [420 40 425 40 + 40 * m]);
    add_line(mdl, 'Controller/1', 'ActDemux/1');
    for k = 1:m
        add_block('simulink/Sinks/Out1', [mdl '/' actNames{k}], 'Position', [500 40 + 40 * (k - 1) 530 54 + 40 * (k - 1)]);
        add_line(mdl, sprintf('ActDemux/%d', k), [actNames{k} '/1']);
    end
end
desc = 'Controleur extrait de la politique apprise par RL Studio (voir equations.txt).';
if isfield(spec, 'description') && ~isempty(spec.description), desc = spec.description; end
set_param(mdl, 'Description', desc);
save_system(mdl, outSlx);
close_system(mdl, 0);
end
