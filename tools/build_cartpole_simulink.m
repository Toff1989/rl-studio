function build_cartpole_simulink(srcSlx, outSlx)
%BUILD_CARTPOLE_SIMULINK Derive le modele du template `cartpole_simulink` d'un modele tiers sous licence MIT.
%
%   Source : https://github.com/jitendra825/Inverted-Pendulum-Simulink (fichier IP_Nonlinear_model.slx),
%            (c) 2021 Jitendra Singh, licence MIT, commit b34c2f7579d93626e06abe5a337bf8e9f913a04b.
%
%   Modifications apportees (la licence MIT les autorise, a condition de conserver l'avis de copyright, joint au template) :
%     - suppression du sous-systeme « 3D Animation » (blocs Virtual Reality retires de MATLAB R2026b) et des
%       blocs d'affichage (Scope, Mux, Demux, gains de conversion en degres) ;
%     - l'echelon de force devient une constante pilotable `Force_cmd` (variable u_force) : c'est l'action de l'agent ;
%     - les sorties x, theta, x_dot, theta_dot sont observables par RL Studio (terminees par des Terminator) ;
%     - le modele d'origine mesure theta depuis la position PENDANTE (stable) : un bloc Bias (-pi) fournit l'angle
%       par rapport a la VERTICALE (theta = 0 : pendule droit), plus naturel pour l'equilibrage ;
%     - angle et position initiaux deviennent les variables theta0 (ecart a la verticale) et x0, tirables au
%       hasard a chaque episode ; le pendule demarre pres de la verticale ;
%     - diagnostic « boucle algebrique » desactive : elle est intrinseque au couplage chariot/pendule et resolue
%       par Simulink (sinon un avertissement a chaque simulation) ;
%     - parametres physiques (m, M, I, l, g, b, c, valeurs de Init_Setup_LQRArd.m) places dans le workspace du modele,
%       qui devient autonome ; solveur ode4, pas fixe de 0,01 s.
%
%   Usage :  build_cartpole_simulink('IP_Nonlinear_model.slx', 'pendulum.slx')
if nargin < 2, outSlx = 'pendulum.slx'; end
[~, srcName] = fileparts(srcSlx);
[~, outName] = fileparts(outSlx);
if bdIsLoaded(srcName), close_system(srcName, 0); end
load_system(srcSlx);
m = srcName;
sub = [m '/Inverted Pendulum System'];

% 1. ne garder que le sous-systeme dynamique
blocks = find_system(m, 'SearchDepth', 1, 'Type', 'Block');
for k = 1:numel(blocks)  % supprimer un bloc supprime aussi ses lignes
    if ~strcmp(blocks{k}, m) && ~strcmp(blocks{k}, sub)
        delete_block(blocks{k});
    end
end
lines = find_system(m, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
while ~isempty(lines)  % lignes restantes : les poignees changent apres chaque suppression, on les relit
    delete_line(lines(1));
    lines = find_system(m, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
end

% 2. action : constante pilotable ; observations : sorties du sous-systeme terminees proprement
add_block('simulink/Sources/Constant', [m '/Force_cmd'], 'Value', 'u_force', 'Position', [60 100 140 130]);
add_line(m, 'Force_cmd/1', 'Inverted Pendulum System/1');
set_param(sub, 'Position', [220 60 420 200]);
for k = 1:4
    t = sprintf('%s/Term%d', m, k);
    add_block('simulink/Sinks/Terminator', t, 'Position', [500 50+k*30 520 70+k*30]);
    if k == 2  % angle par rapport a la verticale : theta_vertical = theta_modele - pi
        add_block('simulink/Math Operations/Bias', [m '/Angle_up'], 'Bias', '-pi', 'Position', [440 82 480 108]);
        add_line(m, 'Inverted Pendulum System/2', 'Angle_up/1');
        add_line(m, 'Angle_up/1', 'Term2/1');
    else
        add_line(m, sprintf('Inverted Pendulum System/%d', k), sprintf('Term%d/1', k));
    end
end

% 3. etat initial pilotable
set_param([sub '/theta_pos'], 'InitialCondition', 'pi + theta0');  % pi = verticale dans le modele d'origine
set_param([sub '/X_pos'], 'InitialCondition', 'x0');

% 4. parametres physiques dans le workspace du modele (valeurs de Init_Setup_LQRArd.m)
mws = get_param(m, 'ModelWorkspace');
params = struct('m', 0.1, 'M', 0.136, 'I', 0.0007176, 'l', 0.2, 'g', 9.81, 'b', 0.00007892, 'c', 0.63, ...
                'theta0', 0.1, 'x0', 0, 'u_force', 0);
for f = fieldnames(params)', mws.assignin(f{1}, params.(f{1})); end

% 5. solveur et documentation
set_param(m, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', '0.01', 'StopTime', '10', 'AlgebraicLoopMsg', 'none');
set_param(m, 'Description', ['Pendule inverse sur chariot. Derive de IP_Nonlinear_model.slx de Jitendra Singh ' ...
    '(MIT, https://github.com/jitendra825/Inverted-Pendulum-Simulink). Voir LICENSE-modele.txt et build_cartpole_simulink.m.']);

% 6. enregistrement (exporte pour une version plus ancienne de MATLAB quand c'est possible)
% ExportToVersion exige un fichier source different de la cible : on passe par un fichier intermediaire
outDir = fileparts(outSlx);
if isempty(outDir), outDir = pwd; end
tmpName = [outName '_full'];
tmpFile = fullfile(outDir, [tmpName '.slx']);
save_system(m, tmpFile);   % « Enregistrer sous » : le modele en memoire s'appelle maintenant tmpName
try
    save_system(tmpName, outSlx, 'ExportToVersion', 'R2024b_SLX');
    fprintf('Exporte pour MATLAB R2024b : %s\n', outSlx);
catch err
    fprintf('Export R2024b impossible (%s) : modele enregistre au format courant.\n', err.message);
    save_system(tmpName, outSlx);
end
if bdIsLoaded(tmpName), close_system(tmpName, 0); end
if bdIsLoaded(outName), close_system(outName, 0); end
delete(tmpFile);
end
