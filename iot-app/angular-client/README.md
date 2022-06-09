# IotApp

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 12.2.6.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

## Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory.

## Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

## Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via a platform of your choice. To use this command, you need to first add a package that implements end-to-end testing capabilities.

## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI Overview and Command Reference](https://angular.io/cli) page.



node set-env.js --env=prod is provided by defalut in package.json so when npm start is called it will generate exposed angular variables

nodemon server has to be called on a separate terminal

default dockerfile redis koy oradan kalksın

triplet projesi de ayrı dosyada olacak absolute path olmasın asla hiçbir yerde nasıl çalışacağını göster

if calling ng serve manually be sure to call node set-env.js --env=prod beforehand to set angular environmental variables from .env file
