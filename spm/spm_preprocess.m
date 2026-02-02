function spm_preprocess()
% ------------------------------------------------------------
% SPM25 preprocessing for T1 + FDG-PET
% Single-subject, no GUI, batch mode
%
% Designed to be called from Python via:
% matlab -batch "spm_preprocess"
% ------------------------------------------------------------

%% ---------- INITIALIZATION ----------
spm('Defaults','fmri');
spm_jobman('initcfg');

root_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(root_dir);

data_dir   = fullfile(project_dir, 'data', 'nifti');
out_dir    = fullfile(project_dir, 'spm', 'outputs');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

t1_file  = fullfile(data_dir, 'T1.nii.gz');
pet_file = fullfile(data_dir, 'PET.nii.gz');

assert(exist(t1_file, 'file') == 2, 'T1 file not found');
assert(exist(pet_file,'file') == 2, 'PET file not found');

%% ---------- UNZIP NIFTI ----------
gunzip(t1_file, out_dir);
gunzip(pet_file, out_dir);

t1_nii  = fullfile(out_dir, 'T1.nii');
pet_nii = fullfile(out_dir, 'PET.nii');

%% ---------- STEP 1: SEGMENTATION (T1) ----------
matlabbatch{1}.spm.spatial.preproc.channel.vols = {t1_nii};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];

tpm_path = fullfile(spm('Dir'),'tpm','TPM.nii');

ngaus = [1 1 2 3 4 2];

for k = 1:6
    matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = ...
        {[tpm_path ',' num2str(k)]};
    matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = ngaus(k);
    matlabbatch{1}.spm.spatial.preproc.tissue(k).native = [k <= 3, 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = [0 0];
end


matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];

%% ---------- STEP 2: COREGISTRATION PET → T1 ----------
matlabbatch{2}.spm.spatial.coreg.estwrite.ref = {t1_nii};
matlabbatch{2}.spm.spatial.coreg.estwrite.source = {pet_nii};
matlabbatch{2}.spm.spatial.coreg.estwrite.other = {''};
matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.tol = ...
    [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{2}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

%% ---------- STEP 3: NORMALIZATION TO MNI ----------
deform_field = fullfile(out_dir, 'y_T1.nii');
rpet_nii     = fullfile(out_dir, 'rPET.nii');

matlabbatch{3}.spm.spatial.normalise.write.subj.def = {deform_field};
matlabbatch{3}.spm.spatial.normalise.write.subj.resample = {
    t1_nii
    rpet_nii
};
matlabbatch{3}.spm.spatial.normalise.write.woptions.bb = ...
    [-78 -112 -70
      78  76   85];
matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{3}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{3}.spm.spatial.normalise.write.woptions.prefix = 'w';

%% ---------- RUN ----------
spm_jobman('run', matlabbatch);

disp('SPM preprocessing finished successfully.');
end
