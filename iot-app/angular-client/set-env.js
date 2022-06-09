const writeFile = require('fs').writeFile;
const argv = require('yargs').argv;
require('dotenv').config();

const environment = argv.env;
const isProd = environment === 'prod';

const targetPath = `./src/environments/environment.${environment}.ts`;
const envConfigFile = `\
// ${environment} environment
export const environment = {
  production: ${isProd},
  SERVER_HOST: "${process.env.SERVER_HOST}",
};
`

writeFile(targetPath, envConfigFile, function (err) {
  if (err) {
    console.log(err);
  } else {
    console.log(`Output generated at ${targetPath}:`);
    console.log(envConfigFile);
  }
});
