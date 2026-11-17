#run before running code:
conda activate /home/tolber3/plasticParticleTest/bactTestData/bact_env

#project directory:
cd plasticParticleTest/BML
cd BML

#run dataset formation:
conda activate /home/tolber3/plasticParticleTest/bactTestData/bact_env
cd /home/tolber3/plasticParticleTest/BML
python scripts/form_sets.py --config configs/gpr_config-722D.yaml
python scripts/form_sets.py --config configs/gpr_config-8n9.yaml

python scripts/form_setsV2.py --config configs/gpr_config-1029.yaml


#run train
python src/BactTrain-gpr.py --config configs/gpr_config.yaml
python src/BactTrain-gpr.py --config configs/gpr_configcopy2.yaml

python src/BactTrain-gpr.py --config configs/gpr_configcopy.yaml
python src/BactTrain-gpr.py --config configs/gpr_configcopy4.yaml

python src/BactTrain-gpr.py --config configs/gpr_config-722D.yaml

python src/BactTrain-gpr.py --config configs/gpr_config-8n9.yaml

python src/BactTrain-gpr.py --config configs/gpr_config-1029.yaml  
python src/BactTrain-gpr.py --config configs/gpr_config-1029copy.yaml  
python src/BactTrain-gpr.py --config configs/gpr_config-1029copy2.yaml  
python src/BactTrain-gpr.py --config configs/gpr_config-1029copy3.yaml  


#launch aim
aim up --repo $SCRATCH/aim-repos/Bact --host 0.0.0.0 --port 43700
#go to 
http://localhost:43700/ or  http://0.0.0.0:43700
or (closes when window closes)
ssh -N -L 43700:node-###:43700 tolber3@slogin.palmetto.clemson.edu

#use 
ssh -N -L 43700:node-####:43700 tolber3@slogin.palmetto.clemson.edu
ssh -N -L 43700:node1816:43700 tolber3@slogin.palmetto.clemson.edu


/home/tolber3/plasticParticleTest/bactTestData/bact_env/bin/python /home/tolber3/plasticParticleTest/Extraction/scripts/V1_FSExtraction.py --config /home/tolber3/plasticParticleTest/Extraction/config/ExtConfig.yaml --overwrite


Things to try:
-   datasets where peak from the same measurements are shuffled to form samples (create multiple sets with different/shuffled peaks at each freq used for sample formation)