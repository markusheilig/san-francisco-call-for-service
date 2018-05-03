package controllers

import javax.inject._
import play.api.libs.json.JsObject
import play.api.libs.json.JsBoolean
import play.api.mvc._

/**
 * This controller creates an `Action` to handle HTTP requests to the
 * application's home page.
 */
class HomeController @Inject()(cc: ControllerComponents) extends AbstractController(cc) {

  val machineModel = new MachineModel()

  /**
   * Create an Action to render an HTML page with a welcome message.
   * The configuration in the `routes` file means that this method
   * will be called when the application receives a `GET` request with
   * a path of `/`.
   */
  def index = Action {
    Ok(views.html.index())
  }

  def sf = Action { implicit request =>
    val callType = request.getQueryString("callType").getOrElse("a")
    val priority = request.getQueryString("prio").getOrElse("3")
    val hood = request.getQueryString("hood").getOrElse("Tenderloin")
    val dt = request.getQueryString("dt").getOrElse("09/02/2017 04:30:09 AM")

    val result = machineModel.isLifeThreatening(callType, priority, hood, dt)
    val json = JsObject(Seq(
      "lifeThreatening" -> JsBoolean(result)
    ))
    Ok(json)
  }

}
