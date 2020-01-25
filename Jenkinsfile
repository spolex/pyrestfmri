properties([pipelineTriggers([githubPush()])])
node{
    def img
    docker.withServer("${SERVER}") {

        stage ('Checkout-dev'){
            git branch: 'develop', url: 'https://github.com/spolex/pyrestfmri'
        }

        stage ("Get image"){
            img = docker.image("spolex/pyrestfmri:${IMG_VER}")
         }

        stage ("Run pyrestfmri container"){
            img.run('--name pyrestfmri -v ${DATA_PATH}:/home/elekin/datos \
            --user 1001:1001 \
            -v ${APP_PATH}:/home/elekin/pyrestfmri \
            -v ${RESULTS}:/home/elekin/results', \
            'python /home/elekin/pyrestfmri/${APP} -c /home/elekin/pyrestfmri/conf/${CONFIG_FILE} ${PARAMS}')
         }

    }
}