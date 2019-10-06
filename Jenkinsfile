properties([pipelineTriggers([githubPush()])])
node{
    def img
    docker.withServer("${SERVER}") {

        stage ('Checkout'){
            git branch: 'develop', url: 'https://github.com/spolex/pyrestfmri'
        }

        stage ("Get image"){
            img = docker.image("spolex/pyrestfmri:0.1")
         }
        stage ("Run pyrestfmri container"){
            img.withRun('--name pyrestfmri -v /home/hadoop/nfs-storage/00-DATASOURCES/00-FMRI:/home/elekin/datos \
            -v /home/hadoop/pyrestfmri:/home/elekin/pyrestfmri  \
            -v /home/hadoop/nfs-storage/02-RESULTADOS:/home/elekin/results','python /home/elekin/pyrestfmri/preprocess.py -c /home/elekin/pyrestfmri/conf/config_test.json -p ${PARALLELISM}')
         }
    }
}