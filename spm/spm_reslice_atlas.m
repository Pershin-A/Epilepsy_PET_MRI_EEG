spm('Defaults','PET');
spm_jobman('initcfg');

base = pwd;

ref = fullfile(base,'outputs','wrPET.nii');
src = fullfile(base,'atlas','labels_Neuromorphometrics.nii');

matlabbatch{1}.spm.spatial.coreg.write.ref = {ref};
matlabbatch{1}.spm.spatial.coreg.write.source = {src};
matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0; % nearest neighbour
matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r';

spm_jobman('run',matlabbatch);
