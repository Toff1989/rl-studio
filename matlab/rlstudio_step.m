function outJson = rlstudio_step(mdlPath, inJson, t0, dt, reset)
%RLSTUDIO_STEP Avance un modele Simulink de dt secondes en conservant son etat.
%   inJson : {"variables": {nom: valeur}, "blocks": [{path,param,value}],
%             "signals": [{name,block,port}]}
%   reset  : true => repart de l'etat initial du modele (t = 0)
%   Renvoie un JSON {nom_signal: derniere valeur} pour chaque signal demande.
%   Les signaux sont journalises en memoire (le .slx n'est pas enregistre).
%   Limite : signaux de bus non supportes.
persistent xFinal allVars
if reset || isempty(allVars)
    xFinal = [];
    allVars = struct();
end

[~, mdl] = fileparts(mdlPath);
if ~bdIsLoaded(mdl)
    load_system(mdlPath);
end

in = jsondecode(inJson);

if reset
    sigs = in.signals;
    for k = 1:numel(sigs)
        if iscell(sigs), s = sigs{k}; else, s = sigs(k); end
        ph = get_param(s.block, 'PortHandles');
        set_param(ph.Outport(s.port), 'DataLogging', 'on', ...
            'DataLoggingNameMode', 'Custom', 'DataLoggingName', s.name);
    end
end

si = Simulink.SimulationInput(mdl);

% les variables (etat initial aleatoire, actions) persistent pendant tout l'episode
if isstruct(in.variables)
    f = fieldnames(in.variables);
    for k = 1:numel(f)
        allVars.(f{k}) = in.variables.(f{k});
    end
end
names = fieldnames(allVars);
mws = get_param(mdl, 'ModelWorkspace');
for k = 1:numel(names)
    if hasVariable(mws, names{k})
        si = si.setVariable(names{k}, allVars.(names{k}), 'Workspace', mdl);
    else
        si = si.setVariable(names{k}, allVars.(names{k}));
    end
end

blocks = in.blocks;
for k = 1:numel(blocks)
    if iscell(blocks), b = blocks{k}; else, b = blocks(k); end
    si = si.setBlockParameter(b.path, b.param, num2str(b.value, 17));
end

si = si.setModelParameter('StopTime', num2str(t0 + dt, 17), ...
    'SignalLogging', 'on', 'SignalLoggingName', 'logsout', ...
    'SaveFinalState', 'on', 'FinalStateName', 'rlXFinal', ...
    'SaveCompleteFinalSimState', 'on');
if ~isempty(xFinal)
    si = si.setInitialState(xFinal);
end

out = sim(si);
xFinal = out.rlXFinal;

res = struct();
sigs = in.signals;
for k = 1:numel(sigs)
    if iscell(sigs), s = sigs{k}; else, s = sigs(k); end
    d = out.logsout.getElement(s.name).Values.Data;
    res.(s.name) = squeeze(d(end, :));
end
outJson = jsonencode(res);
end
