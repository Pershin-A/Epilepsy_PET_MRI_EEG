function spm_warp_gm()
% Warp GM (c1T1.nii) to MNI using existing deformation field

spm('Defaults','fmri');
spm_jobman('initcfg');

root_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(root_dir);
out_dir = fullfile(project_dir, 'spm', 'outputs');

gm_file = fullfile(out_dir, 'c1T1.nii');
def_field = fullfile(out_dir, 'y_T1.nii');

assert(exist(gm_file,'file')==2, 'c1T1.nii not found');
assert(exist(def_field,'file')==2, 'y_T1.nii not found');

matlabbatch{1}.spm.spatial.normalise.write.subj.def = {def_field};
matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {gm_file};

matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = ...
    [-78 -112 -70
      78  76   85];
matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 1;
matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

spm_jobman('run', matlabbatch);

disp('GM normalization finished.');
end
