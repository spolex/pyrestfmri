node{
    def img
    def cont
    docker.withServer('tcp://k8.eastonlab.org:2376') {
    stage ("Get image"){
        img = docker.image("spolex/pyrestfmri:0.1")
    }
    stage ("Run pyrestfmri container"){
        cont img.run('--rm --name pyrestfmri -d \
        -v /home/hadoop/nfs-storage/elekin/00-DATASOURCES/00-FMRI:/home/elekin/datos \
        -v /home/hadoop/pyrestfmri:/home/elekin/pyrestfmri \
        -v /home/hadoop/nfs-storage/02-RESULTADOS/02-PREPROCESS:/home/elekin/results'
        )
    }
    }
}