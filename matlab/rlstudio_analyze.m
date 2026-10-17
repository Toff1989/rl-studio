function outJson = rlstudio_analyze(mdlPath, initPath)
%RLSTUDIO_ANALYZE Analyse un modele Simulink existant avant sa conversion pour RL Studio.
%   Renvoie un JSON : entrees (Inport), sorties (Outport, avec le bloc qui les alimente), constantes numeriques, solveur,
%   variables utilisees par le modele (et ou elles sont definies), blocs d'affichage a retirer, avertissements.
%   initPath : script .m ou fichier .mat optionnel, execute dans le workspace de base AVANT l'analyse (parametres du
%              modele definis hors du .slx).
if nargin < 2, initPath = ''; end
run_init(initPath);
[~, mdl] = fileparts(mdlPath);
if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(mdlPath);
warnings = {};

res = struct();
res.model = mdl;

% ---- entrees et sorties du niveau racine
ins = find_system(mdl, 'SearchDepth', 1, 'BlockType', 'Inport');
res.inports = cell(1, numel(ins));
for k = 1:numel(ins)
    b = ins{k};
    dims = get_param(b, 'PortDimensions');
    dt = get_param(b, 'OutDataTypeStr');
    scalar = strcmp(dims, '-1') || strcmp(dims, '1');
    if ~scalar
        warnings{end+1} = sprintf('Entree "%s" : dimensions %s, seules les entrees scalaires sont convertibles.', get_param(b, 'Name'), dims); %#ok<AGROW>
    end
    if ~isempty(strfind(dt, 'Bus:')) %#ok<STREMP>
        scalar = false;
        warnings{end+1} = sprintf('Entree "%s" : signal de bus, non supporte.', get_param(b, 'Name')); %#ok<AGROW>
    end
    res.inports{k} = struct('path', b, 'name', get_param(b, 'Name'), 'port', str2double(get_param(b, 'Port')), ...
                            'dims', dims, 'convertible', scalar);
end
outs = find_system(mdl, 'SearchDepth', 1, 'BlockType', 'Outport');
res.outports = cell(1, numel(outs));
for k = 1:numel(outs)
    b = outs{k};
    src = ''; sport = 1;
    pc = get_param(b, 'PortConnectivity');
    if ~isempty(pc) && ~isempty(pc(1).SrcBlock) && pc(1).SrcBlock > 0
        src = getfullname(pc(1).SrcBlock);
        sport = pc(1).SrcPort + 1;
    end
    dt = get_param(b, 'OutDataTypeStr');
    ok = ~isempty(src) && isempty(strfind(dt, 'Bus:')); %#ok<STREMP>
    if ~isempty(strfind(dt, 'Bus:')) %#ok<STREMP>
        warnings{end+1} = sprintf('Sortie "%s" : signal de bus, non supporte.', get_param(b, 'Name')); %#ok<AGROW>
    end
    res.outports{k} = struct('path', b, 'name', get_param(b, 'Name'), 'source', src, 'source_port', sport, 'observable', ok);
end

% ---- constantes numeriques (candidates a devenir des actions)
cons = find_system(mdl, 'LookUnderMasks', 'none', 'FollowLinks', 'off', 'BlockType', 'Constant');
res.constants = {};
for k = 1:numel(cons)
    b = cons{k};
    v = get_param(b, 'Value');
    num = str2double(v);
    isvar = false;
    if isnan(num)
        try
            val = slResolve(v, b);
            if isnumeric(val) && isscalar(val), num = double(val); isvar = true; end
        catch
        end
    end
    if ~isnan(num)
        res.constants{end+1} = struct('path', b, 'name', get_param(b, 'Name'), 'value', num, 'expression', v, 'is_variable', isvar); %#ok<AGROW>
    end
end

% ---- solveur
res.solver = struct('type', get_param(mdl, 'SolverType'), 'name', get_param(mdl, 'Solver'), ...
                    'step', get_param(mdl, 'FixedStep'), 'stop', get_param(mdl, 'StopTime'));

% ---- variables utilisees par le modele
res.variables = {};
try
    mws = get_param(mdl, 'ModelWorkspace');
    try
        set_param(mdl, 'SimulationCommand', 'update');   % necessaire a la recherche des variables
    catch err
        msgs = {strtok(err.message, newline)};
        for c = 1:numel(err.cause), msgs{end+1} = strtok(err.cause{c}.message, newline); end %#ok<AGROW>
        warnings{end+1} = ['Mise a jour du diagramme impossible (variable manquante ? indiquer le script ou le .mat qui la definit) : ' ...
                           strjoin(unique(msgs, 'stable'), ' | ')]; %#ok<AGROW>
    end
    vars = Simulink.findVars(mdl, 'SearchMethod', 'cached');
    seen = {};
    for k = 1:numel(vars)
        nm = vars(k).Name;
        if any(strcmp(seen, nm)), continue; end
        seen{end+1} = nm; %#ok<AGROW>
        inModel = hasVariable(mws, nm);
        inBase = evalin('base', sprintf('exist(''%s'', ''var'')', nm)) == 1;
        val = []; scalar = false;
        try
            if inModel, val = getVariable(mws, nm); elseif inBase, val = evalin('base', nm); end
            scalar = isnumeric(val) && isscalar(val);
        catch
        end
        item = struct('name', nm, 'in_model', inModel, 'in_base', inBase, 'scalar', scalar, 'value', []);
        if scalar, item.value = double(val); end
        res.variables{end+1} = item; %#ok<AGROW>
        if ~inModel && ~inBase
            warnings{end+1} = sprintf('Variable "%s" utilisee par le modele mais introuvable : indiquer le script ou le .mat qui la definit.', nm); %#ok<AGROW>
        end
    end
catch err
    warnings{end+1} = ['Analyse des variables impossible : ' err.message];
end

% ---- blocs d'affichage a retirer et liens casses
types = {'Scope', 'Display', 'XYGraph'};
rem = {};
for t = 1:numel(types)
    bs = find_system(mdl, 'LookUnderMasks', 'none', 'FollowLinks', 'off', 'BlockType', types{t});
    for k = 1:numel(bs)
        rem{end+1} = struct('path', bs{k}, 'type', types{t}, 'reason', 'affichage'); %#ok<AGROW>
    end
end
broken = find_system(mdl, 'LookUnderMasks', 'none', 'FollowLinks', 'off', 'LinkStatus', 'unresolved');
for k = 1:numel(broken)
    rem{end+1} = struct('path', broken{k}, 'type', 'lien casse', 'reason', 'bibliotheque introuvable'); %#ok<AGROW>
    warnings{end+1} = sprintf('Bloc "%s" : bibliotheque introuvable (lien casse).', broken{k}); %#ok<AGROW>
end
res.removable = rem;
res.warnings = warnings;
outJson = jsonencode(res);
end

function run_init(initPath)
if isempty(initPath), return; end
[~, ~, ext] = fileparts(initPath);
if strcmpi(ext, '.mat')
    evalin('base', sprintf('load(''%s'');', strrep(initPath, '''', '''''')));
else
    evalin('base', sprintf('run(''%s'');', strrep(initPath, '''', '''''')));
end
end
